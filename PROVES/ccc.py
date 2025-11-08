import argparse
import collections
import random
import time
from typing import Deque, NamedTuple, List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

# ----------------------------
# Arguments
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--env-id", type=str, default="lbforaging:Foraging-8x8-2p-3f-v3")
parser.add_argument("--seed", type=int, default=43)
parser.add_argument("--episodes", type=int, default=6000)
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
# Shared RNN Q-network
# ----------------------------
class RNNQ(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h):
        x = torch.relu(self.fc1(x)).unsqueeze(1)  # [B,1,H]
        out, h = self.rnn(x, h)  # [B,1,H]
        q = self.fc2(out.squeeze(1))  # [B,action_dim]
        return q, h

    def init_hidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.rnn.hidden_size, device=args.device)

# ----------------------------
# Observation flattening
# ----------------------------
def flatten_obs(obs) -> np.ndarray:
    if isinstance(obs, np.ndarray):
        return obs.ravel().astype(np.float32)
    if isinstance(obs, (list, tuple)):
        return np.concatenate([flatten_obs(x) for x in obs]).astype(np.float32)
    if isinstance(obs, dict):
        return np.concatenate([flatten_obs(v) for k, v in sorted(obs.items())]).astype(np.float32)
    return np.array([obs], dtype=np.float32)

# ----------------------------
# Shared IQL Trainer
# ----------------------------
class IQLTrainer:
    def __init__(self, env: gym.Env):
        self.env = env
        self.device = torch.device(args.device)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

        # Determine number of agents and obs/action dimensions
        obs, _ = env.reset(seed=args.seed)
        if isinstance(obs, dict):
            self.agent_ids = sorted(list(obs.keys()))
            self.n_agents = len(self.agent_ids)
            self.obs_dim = flatten_obs(list(obs.values())[0]).shape[0]
        elif isinstance(obs, (list, tuple)):
            self.n_agents = len(obs)
            self.agent_ids = list(range(self.n_agents))
            self.obs_dim = flatten_obs(obs[0]).shape[0]
        else:
            self.n_agents = 1
            self.agent_ids = [0]
            self.obs_dim = flatten_obs(obs).shape[0]

        sample_action_space = env.action_space if hasattr(env, "action_space") else None
        self.action_dim = getattr(sample_action_space, "n", 6)

        # Shared network + target
        self.online = RNNQ(self.obs_dim, self.action_dim, args.hidden_size).to(self.device)
        self.target = RNNQ(self.obs_dim, self.action_dim, args.hidden_size).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = optim.Adam(self.online.parameters(), lr=args.lr)
        # Replay buffer: per-agent experiences but same network
        self.replay = [ReplayBuffer(args.buffer_size) for _ in range(self.n_agents)]

        self.total_steps = 0
        self.grad_steps = 0
        self.last_target_update = 0

    def select_action(self, obs: np.ndarray, epsilon: float, h):
        if random.random() < epsilon:
            return random.randrange(self.action_dim), h
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values, h = self.online(obs_tensor, h)
        return int(q_values.argmax(dim=1).item()), h

    def push_transition(self, agent_idx, obs, action, reward, next_obs, done):
        self.replay[agent_idx].push(obs, action, reward, next_obs, done)

    def train_step(self):
        losses = []
        for i in range(self.n_agents):
            if len(self.replay[i]) < args.batch_size:
                losses.append(None)
                continue
            batch = self.replay[i].sample(args.batch_size)
            batch = Transition(*zip(*batch))
            obs_b = torch.tensor(np.stack(batch.obs), dtype=torch.float32, device=self.device)
            actions_b = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
            rewards_b = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
            next_obs_b = torch.tensor(np.stack(batch.next_obs), dtype=torch.float32, device=self.device)
            dones_b = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

            h_online = self.online.init_hidden(args.batch_size)
            h_target = self.target.init_hidden(args.batch_size)
            # DDQN
            q_online_next, _ = self.online(next_obs_b, h_online)
            next_actions = q_online_next.argmax(dim=1, keepdim=True)
            q_target_next, _ = self.target(next_obs_b, h_target)
            q_target_next = q_target_next.gather(1, next_actions)
            q_target = rewards_b + (1 - dones_b) * args.gamma * q_target_next

            q_vals, _ = self.online(obs_b, h_online)
            q_vals = q_vals.gather(1, actions_b)
            loss = nn.functional.mse_loss(q_vals, q_target.detach())

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
            self.optimizer.step()

            self.grad_steps += 1
            if (self.grad_steps - self.last_target_update) >= args.target_update_interval:
                self.target.load_state_dict(self.online.state_dict())
                self.last_target_update = self.grad_steps
            losses.append(loss.item())
        return losses

# ----------------------------
# Training loop
# ----------------------------
def linear_anneal(start, end, step, decay_steps):
    if step >= decay_steps:
        return end
    return start + (end - start) * step / decay_steps

def run():
    if args.use_wandb:
        wandb_run = wandb.init(project=args.wandb_project,
                               name=args.run_name,
                               config=vars(args),
                               reinit=True)

    env = gym.make(args.env_id)
    try:
        env.reset(seed=args.seed)
        env.action_space.seed(args.seed)
    except Exception:
        pass

    trainer = IQLTrainer(env)
    episode_rewards_history = collections.deque(maxlen=100)
    per_agent_rewards_history = [collections.deque(maxlen=100) for _ in range(trainer.n_agents)]

    start_time = time.time()
    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        if isinstance(obs, dict):
            per_agent_obs_raw = [obs[k] for k in trainer.agent_ids]
        elif isinstance(obs, (list, tuple)):
            per_agent_obs_raw = list(obs)
        else:
            per_agent_obs_raw = [obs] * trainer.n_agents
        per_agent_obs = [flatten_obs(o) for o in per_agent_obs_raw]

        ep_rewards = [0.0] * trainer.n_agents
        hidden_states = [trainer.online.init_hidden() for _ in range(trainer.n_agents)]
        step_in_ep = 0
        terminated = False
        truncated = False
        max_steps = args.max_episode_steps

        while True:
            epsilon = linear_anneal(args.epsilon_start, args.epsilon_final,
                                    trainer.total_steps, args.epsilon_decay_steps)

            actions = []
            new_hidden_states = []
            for i in range(trainer.n_agents):
                a, h = trainer.select_action(per_agent_obs[i], epsilon, hidden_states[i])
                actions.append(a)
                new_hidden_states.append(h)
            hidden_states = new_hidden_states

            step_input = {trainer.agent_ids[i]: actions[i] for i in range(trainer.n_agents)} \
                if trainer.agent_ids is not None else actions

            step_result = env.step(step_input)
            if len(step_result) == 5:
                next_obs_raw, reward_raw, terminated, truncated, info = step_result
            else:
                next_obs_raw, reward_raw, done_flag, info = step_result
                terminated = done_flag
                truncated = False

            rewards = [reward_raw[k] for k in trainer.agent_ids] if isinstance(reward_raw, dict) else \
                      list(reward_raw) if isinstance(reward_raw, (list, tuple, np.ndarray)) else \
                      [float(reward_raw)] * trainer.n_agents

            if isinstance(next_obs_raw, dict):
                per_agent_next_raw = [next_obs_raw[k] for k in trainer.agent_ids]
            elif isinstance(next_obs_raw, (list, tuple)):
                per_agent_next_raw = list(next_obs_raw)
            else:
                per_agent_next_raw = [next_obs_raw] * trainer.n_agents
            per_agent_next = [flatten_obs(o) for o in per_agent_next_raw]

            for i in range(trainer.n_agents):
                trainer.push_transition(i, per_agent_obs[i], actions[i], rewards[i],
                                        per_agent_next[i], terminated or truncated)
                ep_rewards[i] += rewards[i]

            trainer.total_steps += 1
            step_in_ep += 1

            if trainer.total_steps >= args.train_start and trainer.total_steps % args.train_freq == 0:
                losses = trainer.train_step()
                if args.use_wandb:
                    for i, l in enumerate(losses):
                        if l is not None:
                            wandb.log({f"train/loss_agent_{i}": l,
                                       "train/epsilon": epsilon,
                                       "train/step": trainer.total_steps})

            per_agent_obs = per_agent_next

            if terminated or truncated or (max_steps is not None and step_in_ep >= max_steps):
                break

        mean_ep_reward = float(np.mean(ep_rewards))
        episode_rewards_history.append(mean_ep_reward)
        for i in range(trainer.n_agents):
            per_agent_rewards_history[i].append(ep_rewards[i])

        if ep % args.log_every_episodes == 0 or ep == 1:
            stats = {"episode": ep,
                     "episode/mean_reward": mean_ep_reward,
                     "episode/steps": step_in_ep,
                     "total_steps": trainer.total_steps,
                     "train/epsilon": epsilon}
            for i in range(trainer.n_agents):
                stats[f"episode/reward_agent_{i}"] = ep_rewards[i]
                stats[f"episode/avg100_agent_{i}"] = float(np.mean(per_agent_rewards_history[i]))
            stats["episode/avg100_mean"] = float(np.mean(episode_rewards_history))
            if args.use_wandb:
                wandb.log(stats)
            print(f"EP {ep:4d} | meanR {mean_ep_reward:6.2f} | avg100 {stats['episode/avg100_mean']:6.2f} "
                  f"| steps {trainer.total_steps} | eps {epsilon:.3f} | t {time.time() - start_time:.1f}s")

        if ep % 500 == 0:
            # save shared model
            torch.save(trainer.online.state_dict(), f"iql_ddqn_shared_ep{ep}.pth")

    print("Training finished. Total steps:", trainer.total_steps)
    if args.use_wandb:
        wandb_run.finish()

if __name__ == "__main__":
    run()
