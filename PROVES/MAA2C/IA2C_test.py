import re
import os
import torch
import gymnasium as gym
import lbforaging
import numpy as np
import matplotlib
matplotlib.use('Agg')  # for headless environments
import matplotlib.pyplot as plt
import imageio

from IA2C import MAX_EPISODES

ENV_CONF = "Foraging-8x8-2p-3f-v3"
CHECKPOINT_DIR = "./checkpoints_ind"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPISODES = 100      # set to 1 for one episode (with GIF)
SAVE_GIF = (NUM_EPISODES == 1)  # automatically make GIF only if one episode
GIF_PATH = "./episode.gif"
FRAME_DIR = "./frames"
# MODEL_EPISODE = MAX_EPISODES  # load last saved model
MODEL_EPISODE = 280000 

def plot_lbf_observation_tuple(obs, env_conf, save_path):
    """Plots one Foraging observation and saves it to an image file."""
    fig, ax = plt.subplots(figsize=(6,6))
    conf = str(env_conf)
    match_size = re.search(r'(\d+)x(\d+)', conf)
    size_x, size_y = int(match_size.group(1)), int(match_size.group(2))
    match_agents = re.search(r'(\d+)p', conf)
    num_agents = int(match_agents.group(1))
    
    if isinstance(obs, tuple) and len(obs) > 0:
        obs_data = obs[0]
    else:
        obs_data = obs
    
    agent_idx = 3 * num_agents
    agents_obs = obs_data[-agent_idx:]
    apples = obs_data[:-agent_idx]

    # Plot setup
    ax.set_xlim(-0.5, size_x - 0.5)
    ax.set_ylim(-0.5, size_y - 0.5)
    ax.set_xticks(np.arange(size_x))
    ax.set_yticks(np.arange(size_y))
    ax.set_xticks(np.arange(size_x + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(size_y + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
    ax.grid(which='major', color='gray', linestyle='-', linewidth=0)

    # Apples
    num_apples = len(apples) // 3
    for i in range(num_apples):
        y, x, level = int(apples[3*i]), int(apples[3*i+1]), apples[3*i+2]
        ax.plot(x, y, marker='o', color="#C53939", markersize=20 * (8 / size_x), zorder=2)
        ax.text(x, y, f"{level}", color='white', ha='center', va='center',
                fontsize=10, fontweight='bold', zorder=3)

    # Agents
    for i in range(num_agents):
        y, x, level = int(agents_obs[3*i]), int(agents_obs[3*i+1]), agents_obs[3*i+2]
        rect = plt.Rectangle((x-0.4, y-0.4), 0.8, 0.8, facecolor='darkgray',
                             edgecolor='black', linewidth=0.5, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y, f"{int(level)}", color='white', ha='center', va='center',
                fontsize=12, fontweight='bold', zorder=3)

    ax.set_title(env_conf)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

class MAA2CAgent:
    """Wrapper for a trained A2C agent."""
    def __init__(self, obs_dim, act_dim, model_path, device='cpu'):
        self.device = device
        from IA2C import ActorCritic  
        self.model = ActorCritic(obs_dim, act_dim).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def get_action(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits, _ = self.model(obs_tensor)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        return action.item()

def evaluate_agents(num_episodes=1, save_gif=False):
    env = gym.make(ENV_CONF)
    num_agents = env.unwrapped.n_agents
    obs_shape = env.observation_space[0].shape[0]
    act_shape = env.action_space[0].n

    # Load all trained agents
    agents = []
    for i in range(num_agents):
        model_path = os.path.join(CHECKPOINT_DIR, f"MAA2C_agent{i}_episode{MODEL_EPISODE}.pth")
        # model_path = os.path.join(CHECKPOINT_DIR, f"agent{i}_episode{MODEL_EPISODE}.pth")

        agent = MAA2CAgent(obs_shape, act_shape, model_path, device=DEVICE)
        agents.append(agent)

    all_rewards = []
    all_steps = []

    # For GIF saving
    if save_gif:
        os.makedirs(FRAME_DIR, exist_ok=True)

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_rewards = np.zeros(num_agents)
        frame_paths = []
        frame_idx = 0
        steps = 0

        while not done:
            actions = [agent.get_action(obs[i].flatten()) for i, agent in enumerate(agents)]
            next_obs, rewards, terminated, truncated, infos = env.step(actions)
            done = np.any(terminated) or np.any(truncated)

            total_rewards += rewards
            steps += 1

            if save_gif:
                frame_path = os.path.join(FRAME_DIR, f"ep{ep}_frame_{frame_idx:03d}.png")
                plot_lbf_observation_tuple(obs, ENV_CONF, frame_path)
                frame_paths.append(frame_path)
                frame_idx += 1

            obs = next_obs

        all_rewards.append(total_rewards)
        all_steps.append(steps)
        # print(f"Episode {ep+1}/{num_episodes} rewards: {total_rewards} || Team reward: {total_rewards.sum():.3f} || Steps: {steps}")

        # Save GIF for single episode
        if save_gif:
            images = [imageio.imread(fp) for fp in frame_paths]
            imageio.mimsave(GIF_PATH, images, duration=0.5)
            print(f"Saved GIF at {GIF_PATH}")

    env.close()

    print(f"{'Episode':<10}{'Agent Rewards':<20}{'Team Reward':<15}{'Steps':<10}")
    print("-"*60)

    for ep_idx, rewards in enumerate(all_rewards):
        team_reward = rewards.sum()
        steps = all_steps[ep_idx]
        rewards_str = " ".join(f"{r:.3f}" for r in rewards)
        print(f"{ep_idx+1:<10}{rewards_str:<25}{team_reward:<15.3f}{steps:<10}")

    # Aggregate statistics
    all_rewards = np.array(all_rewards)
    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    mean_steps = np.mean(all_steps)

    print("\n===== Evaluation Summary =====")
    print(f"Total episodes: {num_episodes}")
    print(f"Mean per-agent rewards: {mean_rewards}")
    print(f"Std per-agent rewards:  {std_rewards}")
    # print(f"Mean team reward:       {mean_rewards.mean():.3f}")
    print(f"Total team reward:       {mean_rewards.sum():.3f}")
    print(f"Mean steps per episode:  {mean_steps:.3f}")


    return {
        "mean_per_agent": mean_rewards,
        "std_per_agent": std_rewards,
        "team_mean": mean_rewards.mean()
    }

# =====================================================
# Run evaluation
# =====================================================
if __name__ == "__main__":
    results = evaluate_agents(num_episodes=NUM_EPISODES, save_gif=SAVE_GIF)
