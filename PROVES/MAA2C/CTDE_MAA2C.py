import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np
import os

# ---------- CONFIG ----------
NUM_AGENTS = 2
ENV_NAME = "lbforaging:Foraging-8x8-2p-3f-v3"
LR = 1e-3
GAMMA = 0.99
ENTROPY_BETA = 0.01
VALUE_LOSS_COEF = 0.5
MAX_EPISODES = 500000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "checkpoints_ctde"
SAVE_INTERVAL = 2000
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- MODELS ----------
class Actor(nn.Module):
    """Per-agent actor-critic style policy network (actor part only used for action logits)."""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
        )
        self.policy = nn.Linear(128, act_dim)

    def forward(self, x):
        x = self.net(x)
        logits = self.policy(x)
        return logits


class CentralCritic(nn.Module):
    """Centralized critic: takes joint observations (concatenated per-agent obs) and returns V(s)."""
    def __init__(self, joint_obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(joint_obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, joint_obs):
        return self.net(joint_obs).squeeze(-1)  # shape (T,)


# ---------- HELPERS ----------
def select_action(actor, obs):
    """
    obs: local observation (numpy / array)
    returns: action (int), log_prob (tensor), entropy (tensor)
    """
    obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
    logits = actor(obs_t)
    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action), dist.entropy()


def compute_returns(rewards, gamma):
    """Compute discounted returns for a single agent's reward sequence (list of floats)."""
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns  # list length T


# ---------- TRAIN ----------
def train():
    # WandB setup
    wandb.init(project="lbf_a2c", config={
        "env": ENV_NAME,
        "lr": LR,
        "gamma": GAMMA,
        "entropy_beta": ENTROPY_BETA,
        "value_loss_coef": VALUE_LOSS_COEF
    })

    env = gym.make(ENV_NAME)
    # env.observation_space is typically a tuple/list of spaces for multi-agent envs
    obs_dim = env.observation_space[0].shape[0]
    act_dim = env.action_space[0].n
    NUM_AGENTS = env.unwrapped.n_agents
    joint_obs_dim = obs_dim * NUM_AGENTS
    # joint_obs_dim = obs_dim

    # actors: one per agent
    actors = [Actor(obs_dim, act_dim).to(DEVICE) for _ in range(NUM_AGENTS)]
    actor_optimizers = [optim.Adam(actor.parameters(), lr=LR) for actor in actors]

    # centralized critic:
    critic = CentralCritic(joint_obs_dim).to(DEVICE)
    critic_optimizer = optim.Adam(critic.parameters(), lr=LR)

    total_rewards = []
    total_steps_per_episode = []

    for episode in range(MAX_EPISODES):
        obs, _ = env.reset()
        episode_rewards = np.zeros(NUM_AGENTS, dtype=float)
        steps_per_episode = 0

        # Buffers
        log_probs_all = [[] for _ in range(NUM_AGENTS)]
        entropies_all = [[] for _ in range(NUM_AGENTS)]
        rewards_all = [[] for _ in range(NUM_AGENTS)]
        joint_obs_all = []  # list of concatenated joint observations per timestep

        done_episode = False
        while True:
            # store joint observation for critic
            joint_obs = np.concatenate([np.array(o).ravel() for o in obs], axis=0)
            joint_obs_all.append(joint_obs)

            # each agent selects action using its local actor
            actions = []
            for i in range(NUM_AGENTS):
                action, log_prob, entropy = select_action(actors[i], obs[i])
                actions.append(action)
                log_probs_all[i].append(log_prob)
                entropies_all[i].append(entropy)

            # step environment with joint action
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            # rewards is list/array with NUM_AGENTS elements
            for i in range(NUM_AGENTS):
                episode_rewards[i] += rewards[i]
                rewards_all[i].append(rewards[i])

            obs = next_obs
            steps_per_episode += 1

            if np.any(terminated) or np.any(truncated):
                break

        # Episode ended. Compute returns and update networks.
        T = len(joint_obs_all)  # number of timesteps in episode

        # compute returns per agent
        returns_per_agent = []
        for i in range(NUM_AGENTS):
            returns = compute_returns(rewards_all[i], GAMMA)
            returns_per_agent.append(returns)  # list length T

        # convert returns_per_agent -> torch tensor shape (NUM_AGENTS, T)
        returns_tensor = torch.tensor(np.array(returns_per_agent), dtype=torch.float32, device=DEVICE)  # (N, T)

        # joint returns (target for centralized critic) = sum over agents for each timestep -> shape (T,)
        joint_returns = returns_tensor.sum(dim=0)  # (T,)

        # prepare joint_obs_tensor for critic: shape (T, joint_obs_dim)
        joint_obs_tensor = torch.tensor(np.array(joint_obs_all), dtype=torch.float32, device=DEVICE)  # (T, joint_obs_dim)

        # compute values from centralized critic
        values = critic(joint_obs_tensor)  # (T,)
        # values is predicted V(s_t) for each timestep

        # Critic update (value loss)
        critic_optimizer.zero_grad()
        value_loss = VALUE_LOSS_COEF * ((values - joint_returns) ** 2).mean()
        value_loss.backward()
        critic_optimizer.step()

        # For each agent compute policy loss using advantage = returns_i - values
        for i in range(NUM_AGENTS):
            # stack log_probs and entropies
            log_probs = torch.stack(log_probs_all[i])  # (T,)
            entropies = torch.stack(entropies_all[i])  # (T,)
            returns_i = torch.tensor(returns_per_agent[i], dtype=torch.float32, device=DEVICE)  # (T,)

            advantage = returns_i - values.detach()  # (T,)

            policy_loss = -(log_probs * advantage).mean() - ENTROPY_BETA * entropies.mean()

            actor_optimizers[i].zero_grad()
            policy_loss.backward()
            actor_optimizers[i].step()

        total_steps_per_episode.append(steps_per_episode)
        total_rewards.append(episode_rewards.sum())

        # wandb logging
        wandb.log({
            "steps_per_episode": steps_per_episode,
            "mean_steps_100": np.mean(total_steps_per_episode[-100:]),
            "avg_reward_100": np.mean(total_rewards[-100:]),
            "episode_reward": episode_rewards.sum(),
            "critic_value_loss": value_loss.item()
        }, step=episode)

        if episode % 10 == 0:
            print(f"Episode {episode}, Steps: {steps_per_episode}, Reward: {episode_rewards}")
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward (mmax = 1): {np.mean(total_rewards[-100:])}")

        if (episode + 1) % SAVE_INTERVAL == 0:
            # save actors and critic
            for idx, actor in enumerate(actors):
                path = os.path.join(SAVE_DIR, f"MAA2C_actor{idx}_episode{episode+1}.pth")
                torch.save(actor.state_dict(), path)
            critic_path = os.path.join(SAVE_DIR, f"MAA2C_central_critic_episode{episode+1}.pth")
            torch.save(critic.state_dict(), critic_path)
            print(f"Saved model checkpoints at episode {episode+1}")

    env.close()
    wandb.finish()


if __name__ == "__main__":
    train()
