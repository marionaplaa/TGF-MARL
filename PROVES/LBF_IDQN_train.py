# ===========================
# Imports and setup
# ===========================
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


# ===========================
# Deep Q-Network (DQN)
# ===========================
class DQN(nn.Module):
    """A simple 3-layer fully connected Deep Q-Network."""
    def __init__(self, obs_dim, act_dim, lr=LR, device=DEVICE):
        super(DQN, self).__init__()
        self.device = device

        self.model = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        if device == 'cuda':
            self.model.cuda()
        
    def forward(self, x):
        x = torch.FloatTensor(x).to(self.device)
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        return self.model(x)
    
    def get_action(self, state, epsilon=0.05):
        if np.random.random() < epsilon:
            return np.random.randint(self.model[-1].out_features)
        else:
            qvals = self.forward(state)
            return torch.argmax(qvals).item()

# ===========================
# Replay Buffer
# ===========================
class ReplayBuffer:
    """Stores past experiences for experience replay."""
    def __init__(self, capacity=MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)
        self.transition = namedtuple('Transition', ['state', 'action', 'reward', 'done', 'next_state'])
    
    def append(self, state, action, reward, done, next_state):
        self.buffer.append(self.transition(state, action, reward, done, next_state))
    
    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idxs]
        return zip(*batch)
    
    def __len__(self):
        return len(self.buffer)

# ===========================
# Independent Q-Learning Agent
# ===========================
class IQLAgent:
    """Independent Q-Learning agent for multi-agent environments."""
    def __init__(self, obs_dim, act_dim, device=DEVICE, lr=LR, gamma=GAMMA, epsilon=EPSILON_START, eps_decay=EPSILON_DECAY):
        self.qnet = DQN(obs_dim, act_dim, lr, device)
        self.buffer = ReplayBuffer()
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.last_loss = None  # Store last loss for logging
    
    def store_transition(self, state, action, reward, done, next_state):
        self.buffer.append(state, action, reward, done, next_state)
    
    def update(self, batch_size=BATCH_SIZE):
        if len(self.buffer) < batch_size:
            return
        states, actions, rewards, dones, next_states = self.buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.qnet.device)
        actions = torch.LongTensor(actions).unsqueeze(-1).to(self.qnet.device)
        rewards = torch.FloatTensor(rewards).to(self.qnet.device)
        dones = torch.BoolTensor(dones).to(self.qnet.device)
        next_states = torch.FloatTensor(next_states).to(self.qnet.device)
        
        qvals = self.qnet(states).gather(1, actions)
        q_next = self.qnet(next_states).max(dim=1)[0].detach()
        q_next[dones] = 0
        target = rewards + self.gamma * q_next
        
        loss = nn.MSELoss()(qvals.squeeze(), target)
        self.qnet.optimizer.zero_grad()
        loss.backward()
        self.qnet.optimizer.step()

        # Store last loss
        self.last_loss = loss.item()

# ===========================
# Multi-Agent IQL Training Loop
# ===========================
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

        # for i in range(num_agents):
        #     wandb.log({f"agent_{i}_loss": agent.last_loss}, step=total_env_steps)

        for i in range(num_agents):
            wandb.log({
                f"agent_{i}_total_reward": total_rewards[i],
                f"agent_{i}_avg_reward": avg_rewards[i],
                f"agent_{i}_epsilon": agents[i].epsilon,
                f"agent_{i}_loss": agents[i].last_loss if agents[i].last_loss is not None else 0.0
            }, step=ep)

        # -------- Save model checkpoints every 1000 episodes --------
        import os
        if ep % 100 == 0:
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

    # ===========================
    # Environment setup
    # ===========================
    env_conf = "Foraging-8x8-2p-3f-v3"
    env = gym.make(env_conf)

    # ===========================
    # WandB initialization
    # ===========================
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
    # ===========================
    # Run training
    # ===========================
    agents, rewards_history = train_iql(env, n_episodes=MAX_EPISODES, device=DEVICE)

    # ===========================
    # Plot learning curves
    # ===========================
    # plt.figure(figsize=(10, 5))
    # for i in range(env.unwrapped.n_agents):
    #     plt.plot(rewards_history[i], label=f"Agent {i}")
    # plt.xlabel("Episode")
    # plt.ylabel("Total Reward")
    # plt.title("Learning Curves")
    # plt.legend()
    # plt.show()

    # ===========================
    # Cleanup
    # ===========================
    wandb.finish()
    env.close()

