FSDP_CONFIG='fsdp_config.yaml'
TRAIN_SCRIPT='sparse_finetuning.py'

WANDB_PROJECT="alexandre-hackathon2024" \
WANDB_RUN_GROUP="chatmusician_pruned50_spft_LR1e-4_E1_GC1_KDunorm" \
accelerate launch --config_file $FSDP_CONFIG --main_process_port 1234 $TRAIN_SCRIPT
