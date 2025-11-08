#!/usr/bin/env python3
"""
Shared network weights + independent optimizers + per-agent replay buffers.
Double DQN updates per agent. Logs to W&B when requested.
"""
import argparse
import collections
import random
import time
import os
from typing import Deque, NamedTuple, List, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False

# ----------------------------
# CLI / hyperparameters
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--env-id", type=str, default="lbforaging:Foraging-8x8-2p-3f-v3")
parser.add_argument("--seed", type=int, default=529989085)
parser.add_argument("--episodes", type=int, default=10000)
parser.add_argument("--max-episode-steps", type=int, default=None)
parser.add_argument("--buffer-size", type=int, default=5000)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lr", type=float, default=0.0003)
parser.add_argument("--target-update-interval", type=int, default=200)
parser.add_argument("--train-start", type=int, default=1000)
parser.add_argument("--train-freq", type=int, default=1)
parser.add_argument("--epsilon-start", type=float, default=1.0)
parser.add_argument("--epsilon-final", type=float, default=0.05)
parser.add_argument("--epsilon-decay-steps", type=int, default=200000)
parser.add_argument("--hidden-size", type=int, default=128)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--use-wandb", action="store_true")
parser.add_argument("--wandb-project", type=str, default="LBF_proves2")
parser.add_argument("--wandb-name", type=str, default=None)
parser.add_argument("--log-every-episodes", type=int, default=10)
parser.add_argument("--save-model-interval-episodes", type=int, default=500)
parser.add_argument("--save-path", type=str, default="models_shared_weights_independent_opts")
parser.add_argument("--action-mask-key", type=str, default="action_mask")
args = parser.parse_args()

# ----------------------------
# Replay Buffer and Transition
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
# Small MLP Q-network
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
    if isinstance(obs, np.ndarray):
        return obs.ravel().astype(np.float32)
    if isinstance(obs, (list, tuple)):
        arrs = [flatten_obs(x) for x in obs]
        return np.concatenate(arrs).astype(np.float32)
    if isinstance(obs, dict):
        # prefer 'observation' key if present (some wrappers use nested dict)
        if "observation" in obs:
            return flatten_obs(obs["observation"])
        items = []
        for k in sorted(obs.keys()):
            if isinstance(k, str) and ("action" in k and "mask" in k):
                continue
            items.append(flatten_obs(obs[k]))
        if len(items) == 0:
            return np.array([], dtype=np.float32)
        return np.concatenate(items).astype(np.float32)
    try:
        return np.array(obs, dtype=np.float32).ravel()
    except Exception:
        return np.array([hash(str(obs)) % 1_000_000 / 1_000_000.0], dtype=np.float32)

# ----------------------------
# Trainer: shared network weights + per-agent optimizers
# ----------------------------
class SharedWeightsIndependentOptsTrainer:
    def __init__(self, env: gym.Env, device: str):
        self.env = env
        self.device = torch.device(device)
        self.seed_all(args.seed)

        # infer number of agents and sample obs
        obs, info = self.env.reset()
        if isinstance(obs, dict):
            vals = list(obs.values())
            if all(isinstance(v, (np.ndarray, list, tuple, dict)) for v in vals) and len(vals) > 1:
                self.agent_ids = sorted(list(obs.keys()))
                self.n_agents = len(self.agent_ids)
                sample_obs = obs[self.agent_ids[0]]
            else:
                self.agent_ids = list(range(1))
                self.n_agents = 1
                sample_obs = obs
        elif isinstance(obs, (list, tuple)):
            self.n_agents = len(obs)
            self.agent_ids = list(range(self.n_agents))
            sample_obs = obs[0]
        else:
            self.n_agents = 1
            self.agent_ids = [0]
            sample_obs = obs

        self.obs_dim = flatten_obs(sample_obs).shape[0]

        # determine action_dim robustly
        self.action_dim = None
        try:
            import gym
            from gym import spaces
            sp = getattr(self.env, "action_space", None)
            if isinstance(sp, spaces.Discrete):
                self.action_dim = sp.n
            elif isinstance(sp, spaces.Dict):
                first_key = sorted(sp.spaces.keys())[0]
                sub = sp.spaces[first_key]
                if isinstance(sub, spaces.Discrete):
                    self.action_dim = sub.n
            elif isinstance(sp, spaces.Tuple):
                sub = sp.spaces[0]
                if isinstance(sub, spaces.Discrete):
                    self.action_dim = sub.n
        except Exception:
            pass
        if self.action_dim is None:
            if isinstance(sample_obs, dict) and args.action_mask_key in sample_obs:
                self.action_dim = int(np.array(sample_obs[args.action_mask_key]).ravel().shape[0])
        if self.action_dim is None:
            self.action_dim = 6

        # Shared networks
        self.online_net = MLPQ(self.obs_dim, self.action_dim, args.hidden_size).to(self.device)
        self.target_net = MLPQ(self.obs_dim, self.action_dim, args.hidden_size).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())

        # Per-agent optimizers that reference the same parameters
        # Note: each optimizer maintains its own state (Adam moments) but operates on same params.
        self.opts = [optim.Adam(self.online_net.parameters(), lr=args.lr) for _ in range(self.n_agents)]

        # Per-agent replay buffers
        self.replays = [ReplayBuffer(args.buffer_size) for _ in range(self.n_agents)]

        # Counters
        self.total_env_steps = 0
        self.grad_steps = 0
        self.last_target_update = 0

    def seed_all(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def select_action(self, flat_obs: np.ndarray, agent_idx: int, epsilon: float, avail_mask: Optional[np.ndarray] = None) -> int:
        if random.random() < epsilon:
            if avail_mask is not None:
                avail_inds = np.nonzero(avail_mask.ravel())[0]
                if len(avail_inds) == 0:
                    return random.randrange(self.action_dim)
                return int(np.random.choice(avail_inds))
            return random.randrange(self.action_dim)
        x = torch.tensor(flat_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.online_net(x)
        q = q.cpu().numpy().ravel()
        if avail_mask is not None:
            q = np.where(avail_mask.ravel(), q, -1e9)
        return int(int(np.argmax(q)))

    def push_transition(self, agent_idx: int, obs, action, reward, next_obs, done):
        self.replays[agent_idx].push(obs, int(action), float(reward), next_obs, bool(done))

    def train_step_agent(self, agent_idx: int, batch_size: int) -> Optional[float]:
        if len(self.replays[agent_idx]) < batch_size:
            return None

        batch = self.replays[agent_idx].sample(batch_size)
        batch = Transition(*zip(*batch))
        obs_b = torch.tensor(np.stack(batch.obs), dtype=torch.float32, device=self.device)
        actions_b = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_b = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_obs_b = torch.tensor(np.stack(batch.next_obs), dtype=torch.float32, device=self.device)
        dones_b = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Double DQN with shared nets
        q_online_next = self.online_net(next_obs_b)            # [B, A]
        next_actions = q_online_next.argmax(dim=1, keepdim=True)# [B,1]
        q_target_next = self.target_net(next_obs_b).gather(1, next_actions)  # [B,1]
        q_target = rewards_b + (1.0 - dones_b) * args.gamma * q_target_next.detach()

        q_vals = self.online_net(obs_b).gather(1, actions_b)
        loss = nn.functional.mse_loss(q_vals, q_target)

        # Use that agent's optimizer (it references the shared net parameters)
        opt = self.opts[agent_idx]
        opt.zero_grad()
        loss.backward()
        # clip grads across shared params
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        opt.step()

        self.grad_steps += 1
        # periodic hard update of the target network
        if (self.grad_steps - self.last_target_update) >= args.target_update_interval:
            self.target_net.load_state_dict(self.online_net.state_dict())
            self.last_target_update = self.grad_steps

        return float(loss.item())

    def save_models(self, base_path: str):
        os.makedirs(base_path, exist_ok=True)
        torch.save(self.online_net.state_dict(), f"{base_path}/online_shared.pth")
        torch.save(self.target_net.state_dict(), f"{base_path}/target_shared.pth")

# ----------------------------
# Training loop
# ----------------------------
def linear_anneal(start, end, step, decay_steps):
    if step >= decay_steps:
        return end
    t = step / float(max(1, decay_steps))
    return start + (end - start) * t

def run():
    if args.use_wandb and _HAS_WANDB:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
    elif args.use_wandb and not _HAS_WANDB:
        print("Warning: wandb requested but not installed. Continue without W&B.")

    env = gym.make(args.env_id)
    try:
        env.reset(seed=args.seed)
        env.action_space.seed(args.seed)
    except Exception:
        pass

    trainer = SharedWeightsIndependentOptsTrainer(env, device=args.device)

    episode_rewards_history = collections.deque(maxlen=100)
    per_agent_rewards_history = [collections.deque(maxlen=100) for _ in range(trainer.n_agents)]

    start_time = time.time()
    for ep in range(1, args.episodes + 1):
        obs, info = env.reset()
        if isinstance(obs, dict):
            if all(k in obs for k in trainer.agent_ids):
                per_agent_obs_raw = [obs[k] for k in trainer.agent_ids]
            else:
                per_agent_obs_raw = [obs] * trainer.n_agents
        elif isinstance(obs, (list, tuple)):
            per_agent_obs_raw = list(obs)
        else:
            per_agent_obs_raw = [obs] * trainer.n_agents

        per_agent_obs = [flatten_obs(o) for o in per_agent_obs_raw]
        ep_rewards = [0.0 for _ in range(trainer.n_agents)]
        terminated = False
        truncated = False
        step_in_ep = 0
        max_steps = args.max_episode_steps

        while True:
            epsilon = linear_anneal(args.epsilon_start, args.epsilon_final, trainer.total_env_steps if hasattr(trainer, 'total_env_steps') else 0, args.epsilon_decay_steps)

            # build avail masks
            avail_masks = []
            for raw in per_agent_obs_raw:
                if isinstance(raw, dict) and args.action_mask_key in raw:
                    mask = np.array(raw[args.action_mask_key], dtype=np.bool_)
                    if mask.ravel().shape[0] != trainer.action_dim:
                        mask = np.resize(mask.ravel(), (trainer.action_dim,))
                    avail_masks.append(mask)
                else:
                    avail_masks.append(np.ones((trainer.action_dim,), dtype=bool))

            actions = []
            for a in range(trainer.n_agents):
                act = trainer.select_action(per_agent_obs[a], a, epsilon, avail_mask=avail_masks[a])
                actions.append(int(act))

            # format actions for env
            if isinstance(obs, dict) and all(k in obs for k in trainer.agent_ids):
                step_input = {trainer.agent_ids[a]: actions[a] for a in range(trainer.n_agents)}
            else:
                step_input = actions

            step_result = env.step(step_input)
            if len(step_result) == 5:
                next_obs_raw, reward_raw, terminated, truncated, info = step_result
            else:
                next_obs_raw, reward_raw, done_flag, info = step_result
                terminated = done_flag
                truncated = False

            # normalize rewards
            if isinstance(reward_raw, dict):
                rewards = [reward_raw[k] for k in trainer.agent_ids]
            elif isinstance(reward_raw, (list, tuple, np.ndarray)):
                rewards = list(reward_raw)
            else:
                rewards = [float(reward_raw) for _ in range(trainer.n_agents)]

            # normalize next obs
            if isinstance(next_obs_raw, dict) and all(k in next_obs_raw for k in trainer.agent_ids):
                per_agent_next_raw = [next_obs_raw[k] for k in trainer.agent_ids]
            elif isinstance(next_obs_raw, (list, tuple)):
                per_agent_next_raw = list(next_obs_raw)
            else:
                per_agent_next_raw = [next_obs_raw] * trainer.n_agents

            per_agent_next = [flatten_obs(o) for o in per_agent_next_raw]

            # store transitions
            for a in range(trainer.n_agents):
                trainer.push_transition(a, per_agent_obs[a], actions[a], float(rewards[a]), per_agent_next[a], bool(terminated or truncated))
                ep_rewards[a] += float(rewards[a])

            trainer.total_env_steps = getattr(trainer, 'total_env_steps', 0) + 1
            step_in_ep += 1

            # training (per agent)
            if (trainer.total_env_steps >= args.train_start) and (trainer.total_env_steps % args.train_freq == 0):
                for a in range(trainer.n_agents):
                    loss = trainer.train_step_agent(a, args.batch_size)
                    if loss is not None:
                        if args.use_wandb and _HAS_WANDB:
                            wandb.log({f"train/loss_agent_{a}": loss, "train/epsilon": epsilon, "train/step": trainer.total_env_steps})

            per_agent_obs_raw = per_agent_next_raw
            per_agent_obs = per_agent_next

            if terminated or truncated:
                break
            if (max_steps is not None) and (step_in_ep >= max_steps):
                break

        # end episode logging
        mean_ep_reward = float(np.mean(ep_rewards))
        episode_rewards_history.append(mean_ep_reward)
        for a in range(trainer.n_agents):
            per_agent_rewards_history[a].append(ep_rewards[a])

        if ep % args.log_every_episodes == 0 or ep == 1:
            stats = {
                "episode": ep,
                "episode/mean_reward": mean_ep_reward,
                "episode/steps": step_in_ep,
                "total_env_steps": trainer.total_env_steps,
                "train/grad_steps": trainer.grad_steps,
                "train/epsilon": linear_anneal(args.epsilon_start, args.epsilon_final, trainer.total_env_steps, args.epsilon_decay_steps),
                "episode/avg100_mean": float(np.mean(episode_rewards_history)) if len(episode_rewards_history) > 0 else 0.0
            }
            for a in range(trainer.n_agents):
                stats[f"episode/reward_agent_{a}"] = ep_rewards[a]
                stats[f"episode/avg100_agent_{a}"] = float(np.mean(per_agent_rewards_history[a])) if len(per_agent_rewards_history[a]) > 0 else 0.0
                stats[f"replay_size_agent_{a}"] = len(trainer.replays[a])

            print(f"EP {ep:4d} | meanR {mean_ep_reward:6.2f} | avg100 {stats['episode/avg100_mean']:6.2f} | steps {trainer.total_env_steps} | eps {stats['train/epsilon']:.3f} | t {time.time()-start_time:.1f}s")
            if args.use_wandb and _HAS_WANDB:
                wandb.log(stats)

        if ep % args.save_model_interval_episodes == 0:
            trainer.save_models(args.save_path)
            if args.use_wandb and _HAS_WANDB and wandb.run is not None:
                wandb.save(f"{args.save_path}/online_shared.pth")
                wandb.save(f"{args.save_path}/target_shared.pth")

    print("Training finished. Total env steps:", trainer.total_env_steps)
    if args.use_wandb and _HAS_WANDB:
        wandb.finish()

if __name__ == "__main__":
    run()
