from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import re
from tqdm import tqdm
from string import Template
from sparseml.transformers.finetune.data import TextGenerationDataset
from sparseml.transformers.finetune.data.data_args import DataTrainingArguments
from symusic import Score
import os

prompt_template = Template("Human: ${inst} </s> Assistant: ")

model_id = "/network/damian/hackathon/chatmusician_pruned50_oneshot" #"m-a-p/ChatMusician"
midi_output_path = "./midi_data_oneshot"
if not os.path.exists(midi_output_path):
    os.makedirs(midi_output_path)
num_samples = 500
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda", resume_download=True).eval()
model.resize_token_embeddings(len(tokenizer))
model.to(torch.float16)

generation_config = GenerationConfig(
    temperature=0.2,
    top_k=40,
    top_p=0.9,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.1,
    min_new_tokens=10,
    max_new_tokens=512
)

data_args = DataTrainingArguments(
    dataset="json",
    dataset_path="/network/damian/hackathon/dataset",
    remove_columns=["id", "src"],
    max_seq_length=512,
    pad_to_max_length=False
)

dataset_manager = TextGenerationDataset.load_from_registry(
    "custom",
    data_args=data_args,
    split="test",
    tokenizer=tokenizer
)
raw_dataset = dataset_manager.get_raw_dataset()

prompts = [sample["prompt"] for sample in raw_dataset][:num_samples]

num_processed = len(prompts)
num_failed_parse = 0
num_failed_score = 0

idx = 0
for prompt in tqdm(prompts):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    response = model.generate(
            input_ids=inputs["input_ids"].to(model.device),
            attention_mask=inputs['attention_mask'].to(model.device),
            eos_token_id=tokenizer.eos_token_id,
            generation_config=generation_config,
            )
    response = tokenizer.decode(response[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    abc_pattern = r'(X:\d+\n(?:[^\n]*\n)+)'
    regex = re.findall(abc_pattern, response+'\n')
    if len(regex) == 0:
        print("regex error")
        num_failed_parse += 1
        continue
    abc_notation = regex[0]
    try:
        s = Score.from_abc(abc_notation)
        s.dump_midi(os.path.join(midi_output_path, f"sample_{idx}.midi"))
    except:
        print("score conversion error")
        num_failed_score += 1
        continue
    idx += 1

print(f"failed: {num_failed_parse} parsing {num_failed_score} score out of {num_processed}")