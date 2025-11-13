
import lbforaging
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import wandb

from LBF_constants import *
'''LR = 1e-3
MEMORY_SIZE = 5000
MAX_EPISODES = 10000
EPSILON_START = 1.0
EPSILON_DECAY = 0.9995
EPSILON_MIN = 0.05
GAMMA = 0.99
BATCH_SIZE = 64
BURN_IN = 4000
DEVICE = 'cpu'''



class DQN(nn.Module):
    """A simple 3-layer fully connected Deep Q-Network."""
    def __init__(self, obs_dim, act_dim, device="cpu"):
        super(DQN, self).__init__()
        self.device = torch.device(device)
        self.act_dim = act_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )

        self.to(self.device)

    def forward(self, x):
        # accept numpy array or torch tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        else:
            x = x.to(self.device).float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.view(x.size(0), -1)
        return self.net(x)

    def get_action(self, state, epsilon=0.05):
        # return int (action index)
        if np.random.random() < epsilon:
            return np.random.randint(self.act_dim)
        else:
            self.eval()
            with torch.no_grad():
                qvals = self.forward(state)    # shape (1, A)
                action = int(qvals.argmax(dim=1).item())
            self.train()
            return action


class ReplayBuffer:
    """Stores past experiences for experience replay."""
    def __init__(self, capacity=MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)
        self.transition = namedtuple('Transition', ['state', 'action', 'reward', 'done', 'next_state'])
    
    def append(self, state, action, reward, done, next_state):
        self.buffer.append(self.transition(state, action, reward, done, next_state))
    
    # def sample(self, batch_size):
    #     idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
    #     batch = [self.buffer[i] for i in idxs]
    #     return zip(*batch)

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idxs]
        states = np.stack([b.state for b in batch])
        actions = np.array([b.action for b in batch])
        rewards = np.array([b.reward for b in batch], dtype=np.float32)
        dones = np.array([b.done for b in batch], dtype=np.uint8)
        next_states = np.stack([b.next_state for b in batch])
        return states, actions, rewards, dones, next_states

        
    def __len__(self):
        return len(self.buffer)


class IQLAgent:
    """Independent Q-Learning agent for multi-agent environments."""
    def __init__(self, obs_dim, act_dim, device="cpu", lr=LR, gamma=GAMMA, epsilon=EPSILON_START, eps_decay=EPSILON_DECAY):
        self.device = torch.device(device)
        self.qnet = DQN(obs_dim, act_dim, device=device)
        self.target_qnet = DQN(obs_dim, act_dim, device=device)
        self.target_qnet.load_state_dict(self.qnet.state_dict())
        self.target_qnet.eval()

        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)

        self.buffer = ReplayBuffer()
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.last_loss = None
        self.update_count = 0

    def soft_update_target(self, tau=0.01):
        for targ, src in zip(self.target_qnet.parameters(), self.qnet.parameters()):
            targ.data.copy_(targ.data * (1.0 - tau) + src.data * tau)

    def store_transition(self, state, action, reward, done, next_state):
        self.buffer.append(state, action, reward, done, next_state)

    def update(self, batch_size=BATCH_SIZE, grad_clip=None):
        if len(self.buffer) < batch_size:
            return
        states, actions, rewards, dones, next_states = self.buffer.sample(batch_size)
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().unsqueeze(1).to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)

        q_vals = self.qnet(states).gather(1, actions).squeeze(1)  # (B,)
        with torch.no_grad():
            q_next = self.target_qnet(next_states).max(dim=1)[0]  # (B,)
        target = rewards + self.gamma * q_next * (1.0 - dones)

        loss = nn.MSELoss()(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), grad_clip)
        self.optimizer.step()

        self.last_loss = float(loss.item())

        self.update_count += 1

        self.soft_update_target(tau=0.01)

def run_test_episode(env, agents, device="cpu", render=False):
    """Run a single greedy (epsilon=0) evaluation episode."""
    num_agents = env.unwrapped.n_agents
    obs, _ = env.reset()
    done = False
    total_rewards = [0.0] * num_agents

    while not done:
        actions = [agent.qnet.get_action(obs[i].flatten(), epsilon=0.0) for i, agent in enumerate(agents)]
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        done = np.any(terminated) or np.any(truncated)

        for i in range(num_agents):
            total_rewards[i] += rewards[i]

        obs = next_obs

    return total_rewards


def train_iql(env, n_episodes=MAX_EPISODES, device=DEVICE):
    num_agents = env.unwrapped.n_agents
    agents = []

    for i in range(num_agents):
        obs_dim = np.prod(env.observation_space[i].shape)
        act_dim = env.action_space[i].n
        agents.append(IQLAgent(obs_dim, act_dim, device=device))
        print(f"Initialized Agent {i}: obs_dim={obs_dim}, act_dim={act_dim}")
    
    rewards_history = [[] for _ in range(num_agents)]

    # ----------- Burn-in Phase -----------
    print("Burn-in buffer with random actions...")
    steps = 0
    while steps < BURN_IN:
        print(f"Burn-in steps: {steps}/{BURN_IN}", end='\r')
        obs, _ = env.reset()
        done = False
        while not done:
            actions = [env.action_space[i].sample() for i in range(num_agents)]
            next_obs, rewards, terminated, truncated, infos = env.step(actions)
            done = np.any(terminated) or np.any(truncated)

            for i, agent in enumerate(agents):
                agent.store_transition(obs[i].flatten(), actions[i], rewards[i], done, next_obs[i].flatten())
            
            obs = next_obs
            steps += 1
    print("Burn-in complete.\nStarting training...")

    total_env_steps = 0
    # ----------- Main Training Loop -----------
    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        done = False
        total_rewards = [0] * num_agents

        while not done:
            actions = []
            for i, agent in enumerate(agents):
                a = agent.qnet.get_action(obs[i].flatten(), epsilon=agent.epsilon)
                actions.append(a)

            next_obs, rewards, terminated, truncated, infos = env.step(actions)
            done = np.any(terminated) or np.any(truncated)
            total_env_steps += 1

            for i, agent in enumerate(agents):
                agent.store_transition(obs[i].flatten(), actions[i], rewards[i], done, next_obs[i].flatten())
                agent.update()
                total_rewards[i] += rewards[i]

            obs = next_obs

        # Decay epsilon
        for agent in agents:
            agent.epsilon = max(EPSILON_MIN, agent.epsilon * agent.eps_decay)

        # -------- Logging --------
        avg_rewards = [np.mean(rewards_history[i][-100:] + [total_rewards[i]]) for i in range(num_agents)]
        if ep % 10 == 0:
            print(f"Episode {ep}, Avg rewards: {avg_rewards}")


        for i in range(num_agents):
            wandb.log({
                f"agent_{i}_total_reward": total_rewards[i],
                f"agent_{i}_avg_reward": avg_rewards[i],
                f"agent_{i}_epsilon": agents[i].epsilon,
                f"agent_{i}_loss": agents[i].last_loss if agents[i].last_loss is not None else 0.0
            })

        if ep % 500 == 0:
            test_rewards = run_test_episode(env, agents, device=device)
            print(f"[TEST] Episode {ep}: mean test reward = {np.mean(test_rewards):.2f}, per agent = {test_rewards}")
            for i, r in enumerate(test_rewards):
                wandb.log({f"agent_{i}_test_reward": r, "episode": ep})


        import os
        if ep % 500 == 0:
            save_dir = "./checkpoints"
            os.makedirs(save_dir, exist_ok=True)
            for i, agent in enumerate(agents):
                save_path = os.path.join(save_dir, f"agent_{i}_ep{ep}.pt")
                torch.save(agent.qnet.state_dict(), save_path)
                print(f"Saved checkpoint: {save_path}")

        for i in range(num_agents):
            rewards_history[i].append(total_rewards[i])

    return agents, rewards_history

if __name__ == "__main__":


    env_conf = "Foraging-8x8-2p-4f-v3"
    env = gym.make(env_conf)


    run = wandb.init(
        project="LBF_proves",
        config={
            "env_name": env_conf,
            'learning_rate': LR,
            'memory_size': MEMORY_SIZE,
            'max_episodes': MAX_EPISODES,
            'epsilon_start': EPSILON_START,
            'epsilon_decay': EPSILON_DECAY,
            'epsilon_min': EPSILON_MIN,
            'gamma': GAMMA,
            'batch_size': BATCH_SIZE,
            'burn_in': BURN_IN,
            
        },
        sync_tensorboard=True,
        save_code=True,
    )

    agents, rewards_history = train_iql(env, n_episodes=MAX_EPISODES, device=DEVICE)

   
    wandb.finish()
    env.close()

