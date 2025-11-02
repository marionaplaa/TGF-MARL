import re
import os
import torch
import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg')  # <-- This is key for headless environments
import matplotlib.pyplot as plt

import imageio

from LBF_IDQN_train import IQLAgent

# ===========================
# Function to plot observation and save as PNG
# ===========================
def plot_lbf_observation_tuple(obs, env_conf, save_path):
    fig, ax = plt.subplots(figsize=(6,6))
    
    # Parse environment configuration
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
    agents = obs_data[-agent_idx:]
    apples = obs_data[:-agent_idx]

    # Set limits and grid
    ax.set_xlim(-0.5, size_x - 0.5)
    ax.set_ylim(-0.5, size_y - 0.5)
    ax.set_xticks(np.arange(size_x))
    ax.set_yticks(np.arange(size_y))
    ax.set_xticks(np.arange(size_x + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(size_y + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
    ax.grid(which='major', color='gray', linestyle='-', linewidth=0)

    # Plot apples
    num_apples = len(apples) // 3
    for i in range(num_apples):
        y = int(apples[3*i])
        x = int(apples[3*i + 1])
        level = apples[3*i + 2]
        ax.plot(x, y, marker='o', color="#C53939", markersize=20 * (8 / size_x), zorder=2)
        ax.text(x, y, f"{level}", color='white', ha='center', va='center', fontsize=10, fontweight='bold', zorder=3)

    # Plot agents
    for i in range(num_agents):
        y = int(agents[3*i])
        x = int(agents[3*i + 1])
        level = agents[3*i + 2]
        rect = plt.Rectangle((x-0.4, y-0.4), 0.8, 0.8, facecolor='darkgray', edgecolor='black', linewidth=0.5, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y, f"{int(level)}", color='white', ha='center', va='center', fontsize=12, fontweight='bold', zorder=3)

    ax.set_title(env_conf)
    ax.invert_yaxis()
    plt.tight_layout()

    # Save figure
    fig.savefig(save_path)
    plt.close(fig)

# ===========================
# Environment setup
# ===========================
env_conf = "Foraging-8x8-2p-3f-v3"
env = gym.make(env_conf)
num_agents = env.unwrapped.n_agents

# ===========================
# Load agents
# ===========================
agents = []
for i in range(num_agents):
    obs_dim = np.prod(env.observation_space[i].shape)
    act_dim = env.action_space[i].n
    agent = IQLAgent(obs_dim, act_dim, device='cpu')
    
    checkpoint_path = f"./checkpoints/agent_{i}_ep4700.pt"
    agent.qnet.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    agents.append(agent)

# ===========================
# Run episode and save frames
# ===========================
obs, _ = env.reset()
done = False
total_rewards = [0] * num_agents
frame_folder = "./frames"
os.makedirs(frame_folder, exist_ok=True)
frame_paths = []

frame_idx = 0
while not done:
    actions = [agent.qnet.get_action(obs[i].flatten(), epsilon=0.0) for i, agent in enumerate(agents)]
    print(actions)
    next_obs, rewards, terminated, truncated, infos = env.step(actions)
    done = np.any(terminated) or np.any(truncated)

    # Accumulate rewards
    for i in range(num_agents):
        total_rewards[i] += rewards[i]

    # Save plot of observation
    frame_path = os.path.join(frame_folder, f"frame_{frame_idx:03d}.png")
    plot_lbf_observation_tuple(obs, env_conf, frame_path)
    frame_paths.append(frame_path)
    frame_idx += 1

    obs = next_obs

print(f"Total rewards for this episode: {total_rewards}")

# ===========================
# Create GIF
# ===========================
gif_path = "./episode.gif"
images = [imageio.imread(fp) for fp in frame_paths]
imageio.mimsave(gif_path, images, duration=0.5)  # duration in seconds per frame
print(f"Saved episode GIF at {gif_path}")

