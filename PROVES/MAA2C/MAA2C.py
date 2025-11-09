import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np
import os 


NUM_AGENTS = 2
ENV_NAME = "lbforaging:Foraging-8x8-2p-3f-v3"  
LR = 1e-3
GAMMA = 0.99            # discount factor for future rewards
ENTROPY_BETA = 0.01     # entropy regularization term (encourages exploration)
VALUE_LOSS_COEF = 0.5   # coefficient for value loss in total loss
MAX_EPISODES = 300000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "checkpoints"
SAVE_INTERVAL = 2000 

os.makedirs(SAVE_DIR, exist_ok=True)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
        )
        self.policy = nn.Linear(128, act_dim) # output logits for actions
        self.value = nn.Linear(128, 1) # output state value

    def forward(self, x):
        x = self.shared(x)
        return self.policy(x), self.value(x)


def select_action(agent, obs):
    obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
    logits, _ = agent(obs) #compute logits from policy network
    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample() #sample action from distribution. randomness involved
    # chosen action, log prob of action, entropy (uncertainty) of distribution
    return action.item(), dist.log_prob(action), dist.entropy()


def train():
    # WandB setup
    wandb.init(project="maa2c_lbf", config={
        "env": ENV_NAME,
        "lr": LR,
        "gamma": GAMMA,
        "entropy_beta": ENTROPY_BETA,
        "value_loss_coef": VALUE_LOSS_COEF
    })

    # Environment setup
    env = gym.make(ENV_NAME)
    obs_shape = env.observation_space[0].shape[0]
    act_shape = env.action_space[0].n

    NUM_AGENTS = env.unwrapped.n_agents

    # create ActorCritic and optimiser per agent
    agents = [ActorCritic(obs_shape, act_shape).to(DEVICE) for _ in range(NUM_AGENTS)]
    optimizers = [optim.Adam(agent.parameters(), lr=LR) for agent in agents]

    total_rewards = []

    total_steps_per_episode = []
    #training loop
    for episode in range(MAX_EPISODES):
        obs, _ = env.reset()
        episode_rewards = np.zeros(NUM_AGENTS)
        steps_per_episode = 0

        #whole episode info for each agent
        log_probs_all = [[] for _ in range(NUM_AGENTS)]
        rewards_all = [[] for _ in range(NUM_AGENTS)]
        entropies_all = [[] for _ in range(NUM_AGENTS)]
        obs_all = [[] for _ in range(NUM_AGENTS)]

        done = False
        while not done:
            actions = []
            log_probs = []
            entropies = []

            # each agent selects action based on its observation
            for i in range(NUM_AGENTS):
                action, log_prob, entropy = select_action(agents[i], obs[i])
                actions.append(action)
                log_probs.append(log_prob)
                entropies.append(entropy)

            # Step the environment with joint action
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            done = np.logical_or(terminated, truncated)

            for i in range(NUM_AGENTS):
                episode_rewards[i] += rewards[i]
                log_probs_all[i].append(log_probs[i])
                entropies_all[i].append(entropies[i])
                rewards_all[i].append(rewards[i])
                obs_all[i].append(obs[i])

            obs = next_obs
            steps_per_episode += 1

        # EPISODE FINISHED. Compute returns and update
        for i in range(NUM_AGENTS):
            returns = []
            R = 0
            for r in reversed(rewards_all[i]):
                R = r + GAMMA * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

            obs_tensor = torch.tensor(np.array(obs_all[i]), dtype=torch.float32, device=DEVICE)
            logits, values = agents[i](obs_tensor)
            values = values.squeeze(-1)

            advantage = returns - values.detach() # return - baseline
            log_probs = torch.stack(log_probs_all[i])
            entropies = torch.stack(entropies_all[i])

            # Actor: encourage actions with high advantage
            policy_loss = -(log_probs * advantage).mean() - ENTROPY_BETA * entropies.mean()
            # Critic: minimize value estimation error
            value_loss = VALUE_LOSS_COEF * (advantage ** 2).mean()

            loss = policy_loss + value_loss

            optimizers[i].zero_grad()
            loss.backward()
            optimizers[i].step()

        total_steps_per_episode.append(steps_per_episode)
        total_rewards.append(episode_rewards.mean())
        wandb.log({"steps_per_episode": steps_per_episode}, step=episode)
        wandb.log({'mean_steps_per_episode': np.mean(total_steps_per_episode[-100:])}, step=episode)
        wandb.log({"average_total_reward": np.mean(total_rewards[-100:])}, step=episode)
        wandb.log({"episode_reward": episode_rewards.mean(), "episode": episode})
        wandb.log({"total loss": loss.item()}, step=episode)

        if episode % 10 == 0:
            print(f"Episode {episode}, Steps: {steps_per_episode}, Reward: {episode_rewards}")
        if episode % 100 == 0:
            print(f"Episode {episode}, Mean Reward: {np.mean(total_rewards[-100:])}")

        if (episode + 1) % SAVE_INTERVAL == 0:
            for i, agent in enumerate(agents):
                path = os.path.join(SAVE_DIR, f"MAA2C_agent{i}_episode{episode+1}.pth")
                torch.save(agent.state_dict(), path)
            print(f"Saved model checkpoints at episode {episode+1}")

    env.close()
    wandb.finish()


if __name__ == "__main__":
    train()
