from sparseml.transformers import SparseAutoTokenizer
from sparseml.transformers.utils.sparse_tokenizer import SparseAutoTokenizer
from sparseml.transformers.finetune.data import TextGenerationDataset
from sparseml.transformers.finetune.data.data_args import DataTrainingArguments
import argparse
import json

def run(model_path, dataset, num_samples):

    def preprocess(data):
        data["text"] = data["prompt"] + data["text"]
        return data

    tokenizer = SparseAutoTokenizer.from_pretrained(model_path)

    data_args = DataTrainingArguments(
    dataset="json",
    dataset_path=dataset,
    remove_columns=["id", "src"],
    preprocessing_func=preprocess,
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
    tokenized_dataset = dataset_manager.tokenize_and_process(raw_dataset)
    tokenized_dataset = tokenized_dataset[:num_samples]
    with open('calib_quant.json', 'w') as f:
        json.dump(tokenized_dataset, f)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="/network/damian/hackathon/dataset")
    parser.add_argument("--num_samples", type=int, default=1024)
    args = parser.parse_args()
    run(args.model_path, args.dataset, args.num_samples)

