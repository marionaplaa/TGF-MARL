# TGF-MARL

This is the repository of the TFG


MARL 
- https://www.marl-book.com/

Communication related articles
- https://jair.org/index.php/jair/article/view/10952
- https://www.techrxiv.org/doi/full/10.36227/techrxiv.175760290.08279849
- https://openreview.net/forum?id=qpsl2dR9twy
- https://link.springer.com/article/10.1007/s10458-023-09633-6

- https://openreview.net/forum?id=cIrPX-Sn5n
- https://github.com/uoe-agents/epymarl (codi)

Hierarchical rl
- https://arxiv.org/abs/1901.08492
- https://www.cs.toronto.edu/~hinton/absps/dh93.pdf
- https://arxiv.org/abs/1703.01161
- https://arxiv.org/abs/1703.03400


// python3 src/main.py --config=ia2c_ns --env-config=gymma with env_args.time_limit=50 env_args.key="lbforaging:Foraging-15x15-5p-5f-coop-v3"



// python3 src/main.py     --config=ia2c     --env-config=gymma     with env_args.time_limit=50          env_args.key="lbforaging:Foraging-15x15-5p-5f-coop-v3"          evaluate=True          checkpoint_path="results/models/ia2c_seed250291744_lbforagingForaging-15x15-5p-5f-coop-v3_2025-11-15 180831.324893"              test_nepisode=1        render=True


// python3 src/main.py     --config=iql    --env-config=gymma     with env_args.time_limit=50          env_args.key="lbforaging:Foraging-8x8-2p-3f-v3"          evaluate=True          checkpoint_path="results/models/iql_seed343862886_lbforagingForaging-8x8-2p-3f-v3_2025-11-11 214021.871485"              test_nepisode=1        render=True


