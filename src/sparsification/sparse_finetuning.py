from sparseml.transformers.finetune.text_generation import train

model_name = "chatmusician_pruned50_oneshot"
#model_name = "m-a-p/ChatMusician"
distill_teacher = "m-a-p/ChatMusician"
recipe = "distill_recipe.yaml"
dataset_name = "json"
dataset_path = "/network/damian/hackathon/dataset"
remove_columns = ["id", "src"]
#output_dir = "chatmusician_dense_sft_LR1e-6_E1_GC2_KDunorm"
output_dir = "chatmusician_pruned50_soft_LR1e-5_E1_GC2_KDunorm"
overwrite_output_dir = True
num_train_epochs = 1
splits = "train"
precision = "bfloat16"
logging_steps = 25
gradient_checkpointing = True
learning_rate = 0.00001
bf16 = True
max_grad_norm = 2.0 # gradient clipping thresh
lr_scheduler_type = 'cosine'
warmup_ratio = 0.1
report_to = "wandb"
max_seq_len = 512

def custom_mapping(data):
    data["text"] = data["prompt"] + data["text"]
    return data

train(
    model=model_name,
    distill_teacher=distill_teacher,
    recipe=recipe,
    dataset=dataset_name,
    dataset_path=dataset_path,
    preprocessing_func=custom_mapping,
    remove_columns=remove_columns,
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    overwrite_output_dir=overwrite_output_dir,
    splits=splits,
    logging_steps=logging_steps,
    precision=precision,
    gradient_checkpointing=gradient_checkpointing,
    learning_rate=learning_rate,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    lr_scheduler_type=lr_scheduler_type,
    warmup_ratio=warmup_ratio,
    max_seq_length=max_seq_len,
    report_to=report_to,
)