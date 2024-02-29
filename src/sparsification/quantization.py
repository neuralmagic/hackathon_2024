from transformers import AutoTokenizer
import argparse
import json
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

def preprocess(data):
    data["text"] = data["prompt"] + data["text"]
    return data


def quantize(model_path, dataset, bits, group_size, is_marlin_format):

    with open(dataset) as json_data:
        examples = json.load(json_data)
    input_ids = examples['input_ids']
    attention_mask = examples['attention_mask']

    calib = [dict(input_ids=x, attention_mask=y) for (x,y) in zip(input_ids, attention_mask)]

    quantized_model_dir = model_path + "_quant"

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    print(f"Bits: {bits}, group_size: {group_size}, is_marlin_format: {is_marlin_format}")

    quantize_config = BaseQuantizeConfig(
        bits=bits,  # quantize model to 4-bit
        group_size=group_size,
        desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
        is_marlin_format = is_marlin_format
    )

    # load un-quantized model, by default, the model will always be loaded into CPU memory
    model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config)

    # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
    model.quantize(calib)

    # save quantized model using safetensors
    model.save_quantized(quantized_model_dir, use_safetensors=True)


    ## Test inference
    model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0")
    print(tokenizer.decode(model.generate(**tokenizer("auto_gptq is", return_tensors="pt").to(model.device))[0]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="/root/hackathon_2024/calib_quant.json")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--is_marlin_format", action="store_true")
    parser.add_argument("--num_samples", type=int, default=1024)
    args = parser.parse_args()
    quantize(args.model, args.dataset, args.bits, args.group_size, args.is_marlin_format)

