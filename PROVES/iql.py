import argparse
import collections
import math
import random
import time
from typing import Deque, Dict, List, NamedTuple, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

# ----------------------------
# Hyperparameters (changeable)
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--env-id", type=str, default="lbforaging:Foraging-8x8-2p-3f-v3",
                    help="Gymnasium environment id for lbforaging (default: EPyMARL-style ID)")
parser.add_argument("--seed", type=int, default=43)
parser.add_argument("--episodes", type=int, default=6000)
parser.add_argument("--max-episode-steps", type=int, default=None,
                    help="If None use env time_limit or episode termination")
parser.add_argument("--buffer-size", type=int, default=10000)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--target-update-interval", type=int, default=1000,
                    help="Number of gradient steps between target network hard updates")
parser.add_argument("--train-start", type=int, default=1000,
                    help="Number of transitions to collect before starting training")
parser.add_argument("--train-freq", type=int, default=1,
                    help="Train every N environment steps")
parser.add_argument("--epsilon-start", type=float, default=1.0)
parser.add_argument("--epsilon-final", type=float, default=0.05)
parser.add_argument("--epsilon-decay-steps", type=int, default=100000)
parser.add_argument("--hidden-size", type=int, default=128)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--param-sharing", action="store_true",
                    help="If set, share the Q-network across agents (parameter sharing). Default: independent networks.")
parser.add_argument("--wandb-project", type=str, default="iql-ddqn-lbf")
parser.add_argument("--run-name", type=str, default=None)
parser.add_argument("--log-every-episodes", type=int, default=10)
args = parser.parse_args()

# ----------------------------
# Utilities and Replay Buffer
# ----------------------------
Transition = NamedTuple("Transition", [("obs", np.ndarray),
                                       ("action", int),
                                       ("reward", float),
                                       ("next_obs", np.ndarray),
                                       ("done", bool)])

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: Deque[Transition] = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# ----------------------------
# Neural network
# ----------------------------
class MLPQ(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ----------------------------
# Helpers to flatten observations robustly
# ----------------------------
def flatten_obs(obs) -> np.ndarray:
    """
    Accepts different observation formats returned by gym/gymnasium/lbforaging:
      - numpy array -> flatten
      - list/tuple of per-agent observations -> convert to stacked array
      - dict (with agent keys) -> sorts keys and stacks
      - nested dicts -> flattens recursively (concats flattened values)
    Returns a 1D numpy array.
    """
    if isinstance(obs, np.ndarray):
        return obs.ravel().astype(np.float32)
    if isinstance(obs, (list, tuple)):
        arrs = [flatten_obs(x) for x in obs]
        return np.concatenate(arrs).astype(np.float32)
    if isinstance(obs, dict):
        # Sort keys for determinism
        items = []
        for k in sorted(obs.keys()):
            items.append(flatten_obs(obs[k]))
        if len(items) == 0:
            return np.array([], dtype=np.float32)
        return np.concatenate(items).astype(np.float32)
    # fallback: try to convert to array
    try:
        return np.array(obs, dtype=np.float32).ravel()
    except Exception:
        # last resort: string repr
        return np.array([hash(str(obs)) % 1_000_000 / 1_000_000.0], dtype=np.float32)

# ----------------------------
# Core trainer class
# ----------------------------
class IQLTrainer:
    def __init__(self, env: gym.Env, device="cpu"):
        self.env = env
        self.device = device
        self.rng = random.Random(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)

        # Use a sample reset to infer observation/action shapes and number of agents
        obs, info = env.reset()
        # If it's a dict with agent keys, record agent keys.
        self.agent_ids = None
        if isinstance(obs, dict):
            # if mapped by agent id string -> we assume each is the observation for that agent
            # but some gym wrappers return a dict with "observation" or such; detect agent-like keys by heuristic
            # prefer keys that look like agent indices or names
            # if keys are 'agent_0' etc use sorted keys
            # We'll treat dict as per-agent only if each value is array-like
            if all(not isinstance(v, (int, float, list, tuple, np.ndarray)) for v in obs.values()):
                # unlikely; we attempt to flatten entire dict as single global obs -> single-agent fallback
                self.agent_ids = None
                flat = flatten_obs(obs)
                self.n_agents = 1
                self.obs_dim = flat.shape[0]
            else:
                # Treat as per-agent dict
                self.agent_ids = sorted(list(obs.keys()))
                self.n_agents = len(self.agent_ids)
                self.obs_dim = flatten_obs(list(obs.values())[0]).shape[0]
        elif isinstance(obs, (list, tuple)):
            # Per-agent list/tuple
            self.n_agents = len(obs)
            self.agent_ids = list(range(self.n_agents))
            self.obs_dim = flatten_obs(obs[0]).shape[0]
        else:
            # Single-agent observation (unlikely for lbforaging)
            flat = flatten_obs(obs)
            self.n_agents = 1
            self.agent_ids = list(range(self.n_agents))
            self.obs_dim = flat.shape[0]

        # Determine action space per agent (assume identical discrete action spaces)
        sample_action_space = None
        if hasattr(env, "action_space"):
            sample_action_space = env.action_space
        else:
            # try sampling from single-step env after reset
            sample_action_space = env.action_space if hasattr(env, "action_space") else None

        # If action space is a dict/list, try to extract
        if hasattr(sample_action_space, "n"):
            self.action_dim = sample_action_space.n
        else:
            # fallback: try to infer from env metadata or by calling action_space for an agent
            # default to 6 (typical grid actions) if unknown
            self.action_dim = getattr(sample_action_space, "n", 6)

        # Parameter sharing option
        self.param_sharing = args.param_sharing
        print('parameter sharing:', end=' ')
        print(self.param_sharing)

        # Networks, optimizers, buffers: either shared or per-agent
        self.device = torch.device(device)
        if self.param_sharing:
            self.online = MLPQ(self.obs_dim, self.action_dim, args.hidden_size).to(self.device)
            self.target = MLPQ(self.obs_dim, self.action_dim, args.hidden_size).to(self.device)
            self.target.load_state_dict(self.online.state_dict())
            self.opt = optim.Adam(self.online.parameters(), lr=args.lr)
            self.replay = ReplayBuffer(args.buffer_size)
        else:
            self.online = [MLPQ(self.obs_dim, self.action_dim, args.hidden_size).to(self.device)
                           for _ in range(self.n_agents)]
            self.target = [MLPQ(self.obs_dim, self.action_dim, args.hidden_size).to(self.device)
                           for _ in range(self.n_agents)]
            for a in range(self.n_agents):
                self.target[a].load_state_dict(self.online[a].state_dict())
            self.opt = [optim.Adam(self.online[a].parameters(), lr=args.lr) for a in range(self.n_agents)]
            self.replay = [ReplayBuffer(args.buffer_size) for _ in range(self.n_agents)]

        # Counters
        self.total_steps = 0
        self.grad_steps = 0
        self.last_target_update = 0

    def select_action(self, obs: np.ndarray, agent_idx: int, epsilon: float) -> int:
        """Epsilon-greedy; obs is flattened numpy array"""
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.param_sharing:
            with torch.no_grad():
                q = self.online(x)
            return int(q.argmax(dim=1).item())
        else:
            with torch.no_grad():
                q = self.online[agent_idx](x)
            return int(q.argmax(dim=1).item())

    def push_transition(self, agent_idx: int, obs, action, reward, next_obs, done):
        if self.param_sharing:
            self.replay.push(obs, action, reward, next_obs, done)
        else:
            self.replay[agent_idx].push(obs, action, reward, next_obs, done)

    def train_step(self, batch_size: int):
        """One gradient step for each agent (or for shared) using Double DQN logic."""
        if self.param_sharing:
            if len(self.replay) < batch_size:
                return None
            batch = self.replay.sample(batch_size)
            batch = Transition(*zip(*batch))
            obs_b = torch.tensor(np.stack(batch.obs), dtype=torch.float32, device=self.device)
            actions_b = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
            rewards_b = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
            next_obs_b = torch.tensor(np.stack(batch.next_obs), dtype=torch.float32, device=self.device)
            dones_b = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

            # online selects action; target evaluates
            q_online_next = self.online(next_obs_b)  # shape [B, A]
            next_actions = q_online_next.argmax(dim=1, keepdim=True)  # [B,1]
            q_target_next = self.target(next_obs_b).gather(1, next_actions)  # [B,1]
            q_target = rewards_b + (1.0 - dones_b) * args.gamma * q_target_next

            q_vals = self.online(obs_b).gather(1, actions_b)  # [B,1]
            loss = nn.functional.mse_loss(q_vals, q_target.detach())

            self.opt.zero_grad()
            loss.backward()
            # clip grads
            nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
            self.opt.step()

            self.grad_steps += 1
            if (self.grad_steps - self.last_target_update) >= args.target_update_interval:
                self.target.load_state_dict(self.online.state_dict())
                self.last_target_update = self.grad_steps

            return loss.item()

        else:
            losses = []
            for agent_idx in range(self.n_agents):
                if len(self.replay[agent_idx]) < batch_size:
                    losses.append(None)
                    continue
                batch = self.replay[agent_idx].sample(batch_size)
                batch = Transition(*zip(*batch))
                obs_b = torch.tensor(np.stack(batch.obs), dtype=torch.float32, device=self.device)
                actions_b = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
                rewards_b = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
                next_obs_b = torch.tensor(np.stack(batch.next_obs), dtype=torch.float32, device=self.device)
                dones_b = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

                q_online_next = self.online[agent_idx](next_obs_b)
                next_actions = q_online_next.argmax(dim=1, keepdim=True)
                q_target_next = self.target[agent_idx](next_obs_b).gather(1, next_actions)
                q_target = rewards_b + (1.0 - dones_b) * args.gamma * q_target_next

                q_vals = self.online[agent_idx](obs_b).gather(1, actions_b)
                loss = nn.functional.mse_loss(q_vals, q_target.detach())

                self.opt[agent_idx].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.online[agent_idx].parameters(), 10.0)
                self.opt[agent_idx].step()

                self.grad_steps += 1
                if (self.grad_steps - self.last_target_update) >= args.target_update_interval:
                    # update corresponding target
                    self.target[agent_idx].load_state_dict(self.online[agent_idx].state_dict())
                    self.last_target_update = self.grad_steps

                losses.append(loss.item())
            return losses

# ----------------------------
# Training loop
# ----------------------------
def linear_anneal(start, end, step, decay_steps):
    if step >= decay_steps:
        return end
    t = step / float(max(1, decay_steps))
    return start + (end - start) * t

def run():
    # init wandb
    wandb_run = wandb.init(project=args.wandb_project,
                           name=args.run_name,
                           config=vars(args),
                           reinit=True)
    
    # wandb.define_metric("train/step")
    # wandb.define_metric("episode")
    # wandb.define_metric("train/*", step_metric="train/step")
    # wandb.define_metric("episode/*", step_metric="episode")

    env = gym.make(args.env_id)
    # Set seeds if supported
    try:
        env.reset(seed=args.seed)
        env.action_space.seed(args.seed)
    except Exception:
        pass

    trainer = IQLTrainer(env, device=args.device)

    episode_rewards_history = collections.deque(maxlen=100)
    per_agent_rewards_history = [collections.deque(maxlen=100) for _ in range(trainer.n_agents)]

    start_time = time.time()
    for ep in range(1, args.episodes + 1):
        obs, info = env.reset()
        # if env returns dict of agent->obs or list/tuple; normalize to list of per-agent flattened obs
        if trainer.agent_ids is not None and isinstance(obs, dict):
            # get ordered per-agent obs
            per_agent_obs_raw = [obs[k] for k in trainer.agent_ids]
        elif isinstance(obs, (list, tuple)):
            per_agent_obs_raw = list(obs)
        else:
            # single global obs -> duplicate per agent (rare)
            per_agent_obs_raw = [obs] * trainer.n_agents

        per_agent_obs = [flatten_obs(o) for o in per_agent_obs_raw]
        ep_rewards = [0.0 for _ in range(trainer.n_agents)]
        done = False
        truncated = False
        step_in_ep = 0
        max_steps = args.max_episode_steps

        while True:
            # compute decayed epsilon
            epsilon = linear_anneal(args.epsilon_start, args.epsilon_final,
                                    trainer.total_steps, args.epsilon_decay_steps)

            # select actions for all agents
            actions = []
            for a in range(trainer.n_agents):
                act = trainer.select_action(per_agent_obs[a], a, epsilon)
                actions.append(act)

            # Step environment: interpret action format expected by env
            # Common cases:
            #   - env.step takes list/tuple of actions (per-agent)
            #   - env.step takes dict(agent->action)
            #   - env.step takes single action (single-agent)
            step_input = None
            if trainer.agent_ids is not None and isinstance(obs, dict):
                # dict mapping agent id -> action
                step_input = {trainer.agent_ids[a]: actions[a] for a in range(trainer.n_agents)}
            else:
                step_input = actions

            # Gymnasium env.step returns (obs, reward, terminated, truncated, info)
            step_result = env.step(step_input)
            if len(step_result) == 5:
                next_obs_raw, reward_raw, terminated, truncated, info = step_result
            else:
                # older formats
                next_obs_raw, reward_raw, done_flag, info = step_result
                terminated = done_flag
                truncated = False

            # normalize rewards per agent
            if isinstance(reward_raw, dict):
                rewards = [reward_raw[k] for k in trainer.agent_ids]
            elif isinstance(reward_raw, (list, tuple, np.ndarray)):
                rewards = list(reward_raw)
            else:
                # scalar reward -> same for all
                rewards = [float(reward_raw) for _ in range(trainer.n_agents)]

            # normalize next observations
            if trainer.agent_ids is not None and isinstance(next_obs_raw, dict):
                per_agent_next_raw = [next_obs_raw[k] for k in trainer.agent_ids]
            elif isinstance(next_obs_raw, (list, tuple)):
                per_agent_next_raw = list(next_obs_raw)
            else:
                per_agent_next_raw = [next_obs_raw] * trainer.n_agents

            per_agent_next = [flatten_obs(o) for o in per_agent_next_raw]

            # push transitions per agent
            for a in range(trainer.n_agents):
                trainer.push_transition(a, per_agent_obs[a], actions[a], float(rewards[a]),
                                        per_agent_next[a], bool(terminated or truncated))

                ep_rewards[a] += float(rewards[a])

            trainer.total_steps += 1
            step_in_ep += 1

            # training step(s)
            if trainer.param_sharing:
                if len(trainer.replay) >= args.train_start and (trainer.total_steps % args.train_freq == 0):
                    loss = trainer.train_step(args.batch_size)
                    if loss is not None:
                        wandb.log({"train/loss_shared": loss, "train/epsilon": epsilon, "train/step": trainer.total_steps})
            else:
                for a in range(trainer.n_agents):
                    if len(trainer.replay[a]) >= args.train_start and (trainer.total_steps % args.train_freq == 0):
                        losses = trainer.train_step(args.batch_size)
                        # losses is a list with loss or None per agent
                        if isinstance(losses, list):
                            for ai, l in enumerate(losses):
                                if l is not None:
                                    wandb.log({f"train/loss_agent_{ai}": l,
                                               f"train/epsilon": epsilon,
                                               "train/step": trainer.total_steps})

            per_agent_obs = per_agent_next

            if terminated or truncated:
                break
            if (max_steps is not None) and (step_in_ep >= max_steps):
                break

        # end episode
        mean_ep_reward = float(np.mean(ep_rewards))
        episode_rewards_history.append(mean_ep_reward)
        for a in range(trainer.n_agents):
            per_agent_rewards_history[a].append(ep_rewards[a])

        # logging
        if ep % args.log_every_episodes == 0 or ep == 1:
            stats = {"episode": ep,
                     "episode/mean_reward": mean_ep_reward,
                     "episode/steps": step_in_ep,
                     "total_steps": trainer.total_steps,
                     "train/grad_steps": trainer.grad_steps,
                     "train/epsilon": linear_anneal(args.epsilon_start, args.epsilon_final,
                                                    trainer.total_steps, args.epsilon_decay_steps)}
            for a in range(trainer.n_agents):
                stats[f"episode/reward_agent_{a}"] = ep_rewards[a]
                stats[f"episode/avg100_agent_{a}"] = float(np.mean(per_agent_rewards_history[a])) if len(per_agent_rewards_history[a]) > 0 else 0.0
            stats["episode/avg100_mean"] = float(np.mean(episode_rewards_history)) if len(episode_rewards_history) > 0 else 0.0


            stats["episode"] = ep
            wandb.log(stats)

            elapsed = time.time() - start_time
            print(f"EP {ep:4d} | meanR {mean_ep_reward:6.2f} | avg100 {stats['episode/avg100_mean']:6.2f} | steps {trainer.total_steps} | eps {stats['train/epsilon']:.3f} | t {elapsed:.1f}s")

        if ep % 500 == 0:
            # save model checkpoint
            if trainer.param_sharing:
                torch.save(trainer.online.state_dict(), f"iql_ddqn_lbf_shared_ep{ep}.pth")
            else:
                for a in range(trainer.n_agents):
                    torch.save(trainer.online[a].state_dict(), f"iql_ddqn_lbf_agent{a}_ep{ep}.pth")

    print("Training finished. Total steps:", trainer.total_steps)
    wandb_run.finish()


if __name__ == "__main__":
    run()
