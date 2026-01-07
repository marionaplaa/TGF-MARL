# TGF-MARL

This repository contains experiments for multi-agent reinforcement learning (MARL) algorithms, built on top of the **EPyMARL** framework. The experiments are primarily focused on cooperative multi-agent tasks in the **Level-Based Foraging (LBF)** environment.

## Overview

All experiments are built using the [EPyMARL library](https://github.com/uoe-agents/epymarl), a PyTorch-based framework for multi-agent RL research.

## Repository Structure

```
TGF-MARL/
├── epymarl/              # EPyMARL framework (main training code)
│   ├── src/
│   │   ├── main.py       # Entrypoint for experiments
│   │   ├── run.py        # Training loop orchestration
│   │   ├── learners/     # Algorithm implementations (QLearner, etc.)
│   │   ├── controllers/  # Multi-agent controllers (MAC)
│   │   ├── runners/      # Environment interactions
│   │   ├── components/   # Replay buffer, transforms, scheduling
│   │   ├── envs/         # Environment wrappers
│   │   ├── modules/      # Neural network modules (mixers, agents)
│   │   └── utils/        # Utilities (logging, rewards, etc.)
│   ├── requirements.txt  # Python dependencies
└── PROVES/               # Proof-of-concept & experiment configurations
    ├── LBF.ipynb         # Jupyter notebook for LBF environment analysis
    ├── experiments.yaml  # Experiment configuration 
    └── hyperparameter_search/  # Hyperparameter search configs
        ├── search_ia2c.yaml
        ├── search_ia2c_ns.yaml
        ├── search_iql.yaml
        ├── search_iql_ns.yaml
        ├── search_qmix.yaml
        ├── search_vdn.yaml
        ├── search_coma.yaml
        ├── search_maa2c.yaml
        ├── search_maddpg.yaml
        ├── search_mappo.yaml
        └── search_ippo.yaml
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
python3 src/main.py --config=ia2c_ns --env-config=gymma with env_args.time_limit=50 env_args.key="lbforaging:Foraging-8x8-2p-3f-v3"
```
To run a hyperparameter search: 
```bash
python3 search.py run --config=search_ia2c_ns.yaml --seeds 2 locally
```
`--seeds x` determines the number of seeds used with that algorithm. 


### Evaluation 

To evaluate a trained model:

```bash
python3 main.py --config=ia2c_ns --env-config=gymma with env_args.time_limit=50 env_args.key="lbforaging:Foraging-15x15-4p-4f-v3" evaluate=True checkpoint_path="models_videos/ia2c_ns_seed1_lbforaging:Foraging-15x15-4p-4f-v3_2026-01-01 02:32:09.119328" test_nepisode=1
```
The render option can be set in the `default.yaml` file or directly in the command. The config file to be used, the environment configuration and the model path need to be specified. 

## ENV_FILES Folder

The `ENV_FILES` folder contains proof-of-concept experiments and hyperparameter search configurations:

### Files

- **LBF.ipynb**: Jupyter notebook for interactive exploration and visualization of Level-Based Foraging experiments. Used to understand agent's observations, reward function and to try different LBF configurations.

- **experiments.yaml**: Example experiment configuration file showing how to structure multi-run experiment definitions with different seeds and hyperparameters.

- **hyperparameter_search/**: Contains hyperparameter search configurations (YAML files) for each algorithm:
  - `search_ia2c.yaml` - Independent A2C hyperparameter ranges
  - `search_ia2c_ns.yaml` - Independent A2C with non-shared parameters
  - `search_iql.yaml` - Independent Q-Learning hyperparameter ranges
  - `search_iql_ns.yaml` - Independent Q-Learning with non-shared parameters
  - `search_qmix.yaml` - QMIX (value mixing) hyperparameter ranges
  - `search_vdn.yaml` - VDN (value decomposition) hyperparameter ranges
  - `search_coma.yaml` - COMA (counterfactual multi-agent) hyperparameter ranges
  - `search_maa2c.yaml` - Multi-Agent A2C hyperparameter ranges
  - `search_maddpg.yaml` - MADDPG hyperparameter ranges
  - `search_mappo.yaml` - MAPPO hyperparameter ranges
  - `search_ippo.yaml` - Independent PPO hyperparameter ranges

These YAML files define search spaces for algorithms like Ray Tune or similar hyperparameter optimization frameworks.

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


