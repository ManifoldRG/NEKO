## test halfcheetah
python train.py --embed_dim=128 --layers=3 --heads=1 --training_steps=10000 --log_eval_freq=1000 --warmup_steps=100 --batch_size=16 -k=32 --eval_episodes=1 --device=cuda

## test breakout
python train.py --embed_dim=128 --layers=3 --heads=1 --training_steps=10000 --log_eval_freq=1000 --warmup_steps=100 --batch_size=4 -k=512 --eval_episodes=1 --device=cuda --datasets Breakout-expert_s0-v0