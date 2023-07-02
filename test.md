## test halfcheetah
python train.py --embed_dim=128 --layers=3 --heads=1 --training_steps=10000 --log_eval_freq=1000 --warmup_steps=100 --batch_size=16 -k=32 --eval_episodes=1 --device=cuda

## test breakout
python train.py --embed_dim=128 --layers=3 --heads=1 --training_steps=10000 --log_eval_freq=1000 --warmup_steps=100 --batch_size=4 -k=512 --eval_episodes=1 --device=cuda --datasets Breakout-expert_s0-v0

## w/out prompting
python train.py --embed_dim=256 --layers=3 --heads=1 --training_steps=10000 --log_eval_freq=1000 --warmup_steps=1000 --batch_size=8 -k=128 --eval_episodes=10 --promptless_eval --prompt_ep_proportion=0

## Testing FP16
python train.py --embed_dim=768 --layers=6 --heads=4 --training_steps=200000 --log_eval_freq=20000 --warmup_steps=10000 --batch_size=32 -k=240 --eval_episodes=1 --activation_fn=gelu --save_model --save_mode=checkpoint --disable_cosine_decay --datasets d4rl_halfcheetah-expert-v2 --mixed_precision=fp16

## Testing LORA
python train.py --training_steps=200000 --log_eval_freq=20000 --warmup_steps=10000 --batch_size=4 -k=240 --eval_episodes=10 --activation_fn=gelu --save_model --save_mode=checkpoint --disable_cosine_decay --datasets d4rl_halfcheetah-expert-v2 --pretrained_lm=gpt2 --lora