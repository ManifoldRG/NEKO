python -m pdb train.py \
  --embed_dim=768 \
  --layers=6 \
  --heads=24 \
  --training_steps=1000 \
  --log_eval_freq=10 \
  --warmup_steps=10 \
  --batch_size=4 -k=240 \
  --eval_episodes=10 \
  --sequence_length=1024 \
  --activation_fn=gelu \
  --save_model \
  --vqa_prop=1.0 \
  --vqa_dataset='HuggingFaceM4/VQAv2'


