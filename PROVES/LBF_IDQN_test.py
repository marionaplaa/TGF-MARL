# import torch

# # Number of agents
# num_agents = env.unwrapped.n_agents

# # Observation and action dimensions for each agent
# agents = []
# for i in range(num_agents):
#     obs_dim = np.prod(env.observation_space[i].shape)
#     act_dim = env.action_space[i].n
#     agent = IQLAgent(obs_dim, act_dim, device='cpu')  # device can be 'cuda'
    
#     # Load the trained model
#     checkpoint_path = f"./checkpoints/agent_{i}_ep5000.pt"
#     agent.qnet.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    
#     agents.append(agent)

# obs, _ = env.reset()
# done = False
# total_rewards = [0] * num_agents

# while not done:
#     actions = []
#     for i, agent in enumerate(agents):
#         # ε = 0 → fully greedy actions
#         action = agent.qnet.get_action(obs[i].flatten(), epsilon=0.0)
#         actions.append(action)
        
#     print(actions)
    
#     next_obs, rewards, terminated, truncated, infos = env.step(actions)
#     done = np.any(terminated) or np.any(truncated)

#     print(next_obs)
#     # Accumulate rewards
#     for i in range(num_agents):
#         total_rewards[i] += rewards[i]

#     obs = next_obs

# print(f"Total rewards for this episode: {total_rewards}")

'''
N = 10
avg_rewards = np.zeros(num_agents)

for ep in range(N):
    obs, _ = env.reset()
    done = False
    ep_rewards = np.zeros(num_agents)
    
    while not done:
        actions = [agent.qnet.get_action(obs[i].flatten(), epsilon=0.0) for i, agent in enumerate(agents)]
        obs, rewards, terminated, truncated, infos = env.step(actions)
        print(actions)
        done = np.any(terminated) or np.any(truncated)
        ep_rewards += rewards
    
    avg_rewards += ep_rewards

avg_rewards /= N
print(f"Average rewards over {N} episodes: {avg_rewards}")

'''