# TFG-MARL

This repository contains experiments for multi-agent reinforcement learning (MARL) algorithms, built on top of the **EPyMARL** framework. The experiments are primarily focused on cooperative multi-agent tasks in the **Level-Based Foraging (LBF)** environment.

## Overview

All experiments are built using the [EPyMARL library](https://github.com/uoe-agents/epymarl), a PyTorch-based framework for multi-agent RL research.

## Environment: Level-Based Foraging (LBF)

**Level-Based Foraging (LBF)** ([GitHub](https://github.com/semitable/lb-foraging)) is a grid-based multi-agent environment in which agents move across a map to collect food items. Although the task may appear simple, the dynamics of food collection introduce complex coordination challenges. Each agent and each food item is assigned a level, and a food item can only be collected if the sum of the levels of all participating agents is greater than or equal to the level of that food. When a collection occurs, all agents that contributed to the required level receive a reward proportional to their own level.

The environment supports multiple configurations for the grid size, number of agents, number of apples, cooperative setting, and partial observability. In the standard setting, an agent may collect food individually if its level is high enough, which creates competitive incentives. In contrast, the cooperative setting disallows individual collection, forcing agents to coordinate in order to achieve any reward. In the standard setting, the environment has full observability, which means that each agent's observation equals the full state. With the partial observability setting, each agent's view is limited to two cells in every direction, a patch of 5×5 cells.

Agents have six discrete actions: *do nothing*, *load*, and *move* in one of the four cardinal directions (up, down, left, right). Each agent receives observations that encode both food and agent information. For every food item and every agent (including itself), the observation includes the x-position, y-position, and level. Thus, an agent's observation vector consists of the positions and levels of all food items, the agent's own information, and the information of all other agents.

Environment configurations follow the naming scheme: `lbforaging:Foraging-{grid_size}x{grid_size}-{sight}s-{players}p-{food}f-{coop}-v3`

## Repository Structure

```
TFG-MARL/
├── epymarl/              # EPyMARL framework (main training code)
│   ├── src/
│   │   ├── main.py       # Entrypoint for experiments
│   │   ├── run.py        # Training loop orchestration
│   │   ├── search.py     # Hyperparameter search orchestration
│   │   ├── learners/     # Algorithm implementations (QLearner, etc.)
│   │   ├── controllers/  # Multi-agent controllers (MAC)
│   │   ├── runners/      # Environment interactions
│   │   ├── components/   # Replay buffer, transforms, scheduling
│   │   ├── envs/         # Environment wrappers
│   │   ├── modules/      # Neural network modules (mixers, agents)
│   │   ├── utils/        # Utilities (logging, rewards, etc.)
│   │   ├── config/       # Configuration templates
│   │   └── models_videos/ # Saved trained models and videos
│   ├── requirements.txt  # Python dependencies
│   └── results/          # Training results and logs
├── ENV_FILES/            # Experiment configurations & analysis
│   ├── LBF.ipynb         # Jupyter notebook for LBF environment analysis
│   ├── default.yaml      # Default configuration
│   ├── experiments.yaml  # Experiment configuration
│   ├── algs/             # Algorithm-specific configs
│   │   ├── ia2c.yaml
│   │   ├── ia2c_ns.yaml
│   │   ├── iql.yaml
│   │   ├── iql_ns.yaml
│   │   ├── qmix.yaml
│   │   ├── vdn.yaml
│   │   ├── coma.yaml
│   │   ├── maa2c.yaml
│   │   ├── maddpg.yaml
│   │   ├── mappo.yaml
│   │   └── ippo.yaml
│   └── hyperparameter_search/  # Hyperparameter search configs
│       ├── search_ia2c.yaml
│       ├── search_ia2c_ns.yaml
│       ├── search_iql.yaml
│       ├── search_iql_ns.yaml
│       ├── search_qmix.yaml
│       ├── search_vdn.yaml
│       ├── search_coma.yaml
│       ├── search_maa2c.yaml
│       ├── search_maddpg.yaml
│       ├── search_mappo.yaml
│       └── search_ippo.yaml
└── lb-foraging/          # Level-Based Foraging environment
```

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/uoe-agents/epymarl
```

### 2. Install Dependencies

Install the core dependencies:

```bash
pip install -r requirements.txt
```

Install environment dependencies (for LBF, SMAC, Gymnasium, etc.):

```bash
pip install -r env_requirements.txt
```



### Training

Train IA2C with non-shared parameters on LBF:
```bash
cd epymarl
python3 src/main.py --config=ia2c_ns --env-config=gymma with env_args.time_limit=50 env_args.key="lbforaging:Foraging-8x8-2p-3f-v3"
```

To run a hyperparameter search: 
```bash
cd epymarl
python3 src/search.py run --config=search_ia2c_ns.yaml --seeds 2 locally
```

`--seeds x` determines the number of seeds used with that algorithm. 


### Evaluation 

To evaluate a trained model:

```bash
python3 main.py --config=ia2c_ns --env-config=gymma with env_args.time_limit=50 env_args.key="lbforaging:Foraging-15x15-4p-4f-v3" evaluate=True checkpoint_path="models_videos/ia2c_ns_seed1_lbforaging:Foraging-15x15-4p-4f-v3_2026-01-01 02:32:09.119328" test_nepisode=1
```

The render option can be set in the `config/default.yaml` file or directly in the command. The config file to be used, the environment configuration and the model path need to be specified. 

## ENV_FILES Folder

The `ENV_FILES` folder contains experiment configurations and hyperparameter search setup:

### Files

- **LBF.ipynb**: Jupyter notebook for interactive exploration and visualization of Level-Based Foraging experiments. Used to understand agent's observations, reward function and to try different LBF configurations.

- **default.yaml**: Default configuration settings for training runs.

- **experiments.yaml**: Example experiment configuration file showing how to structure multi-run experiment definitions with different seeds and hyperparameters.

- **algs/**: Contains algorithm-specific configuration files for each supported algorithm:
  - `ia2c.yaml` / `ia2c_ns.yaml` - Independent A2C with optional non-shared parameters
  - `iql.yaml` / `iql_ns.yaml` - Independent Q-Learning with optional non-shared parameters
  - `qmix.yaml` - QMIX (value mixing)
  - `vdn.yaml` - VDN (value decomposition)
  - `coma.yaml` - COMA (counterfactual multi-agent)
  - `maa2c.yaml` - Multi-Agent A2C
  - `maddpg.yaml` - MADDPG (Multi-Agent DDPG)
  - `mappo.yaml` - MAPPO (Multi-Agent PPO)
  - `ippo.yaml` - Independent PPO

- **hyperparameter_search/**: Contains hyperparameter search configurations (YAML files) for each algorithm:
  - `search_ia2c.yaml`, `search_ia2c_ns.yaml` - Independent A2C hyperparameter ranges
  - `search_iql.yaml`, `search_iql_ns.yaml` - Independent Q-Learning hyperparameter ranges
  - `search_qmix.yaml` - QMIX hyperparameter ranges
  - `search_vdn.yaml` - VDN hyperparameter ranges
  - `search_coma.yaml` - COMA hyperparameter ranges
  - `search_maa2c.yaml` - Multi-Agent A2C hyperparameter ranges
  - `search_maddpg.yaml` - MADDPG hyperparameter ranges
  - `search_mappo.yaml` - MAPPO hyperparameter ranges
  - `search_ippo.yaml` - Independent PPO hyperparameter ranges

These YAML files define search spaces for hyperparameter optimization frameworks.

## Key EPyMARL Components

### Learners (src/learners/)
- `q_learner.py` - Q-Learning with optional mixing (VDN, QMIX)
- `coma_learner.py` - COMA algorithm
- `actor_critic_learner.py` - Actor-Critic methods
- `ppo_learner.py` - Proximal Policy Optimization
- `maddpg_learner.py` - Multi-Agent DDPG

### Controllers
- Multi-Agent Controllers (MAC) that manage agent neural networks
- `basic_controller.py` - Basic RNN-based controller
- `non_shared_controller.py` - Per-agent independent networks
- `maddpg_controller.py` - MADDPG-specific controller

### Runners 
- `episode_runner.py` - Single-environment episode generation
- `parallel_runner.py` - Parallel environment execution for on-policy algorithms

## Logging & Monitoring
Results are saved to `results/` with the experiment token as directory name.

## Common Tasks




## References

- **EPyMARL**: [GitHub](https://github.com/uoe-agents/epymarl) | [Paper](https://arxiv.org/abs/2006.07869)
- **Level-Based Foraging**: [GitHub](https://github.com/uoe-agents/lb-foraging)

## License

See `epymarl/LICENSE` for details on the EPyMARL framework license.


