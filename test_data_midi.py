import re
from sparseml.transformers.finetune.data import TextGenerationDataset
from sparseml.transformers.finetune.data.data_args import DataTrainingArguments
from symusic import Score
import os

midi_output_path = "./midi_data_ground_truth"
if not os.path.exists(midi_output_path):
    os.makedirs(midi_output_path)
num_samples = 500


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
    tokenizer=None
)
raw_dataset = dataset_manager.get_raw_dataset()

outputs = [sample["text"] for sample in raw_dataset][:num_samples]
num_processed = len(outputs)
num_failed_parse = 0
num_failed_score = 0

idx = 0
for response in outputs:
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