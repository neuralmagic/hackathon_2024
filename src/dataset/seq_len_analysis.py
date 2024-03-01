import os
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy
from matplotlib import pyplot as plt
from tqdm import tqdm

model = "m-a-p/ChatMusician" # for tokenizer
dataset_path = "/network/damian/hackathon/dataset"
split = "train"
output = "musicpile_tokenlen"
data_type = torch.float16

tokenizer = AutoTokenizer.from_pretrained(model)

data = load_dataset("json", data_files=os.path.join(dataset_path, split + ".json"))["train"]

token_len = []

for sample in tqdm(data):
    tokens = tokenizer(sample["prompt"] + sample["text"])["input_ids"]
    token_len.append(len(tokens))

token_len = numpy.array(token_len)

numpy.save(f"{output}.npy", token_len)

below_512 = 100. * numpy.sum(token_len <= 512) / token_len.size
below_1024 = 100. * numpy.sum(token_len <= 1024) / token_len.size
below_2048 = 100. * numpy.sum(token_len <= 2048) / token_len.size

plt.hist(token_len, 100)
plt.axvline(512, color='k', linestyle='dashed', linewidth=1)
plt.text(600, plt.ylim()[1]*0.9, 'Lower than 512: {:.2f}%'.format(below_512))
plt.axvline(1024, color='k', linestyle='dashed', linewidth=1)
plt.text(1100, plt.ylim()[1]*0.8, 'Lower than 1024: {:.2f}%'.format(below_1024))
plt.axvline(2048, color='k', linestyle='dashed', linewidth=1)
plt.text(1300, plt.ylim()[1]*0.7, 'Lower than 2048: {:.2f}%'.format(below_2048))
plt.xlabel("Length of input tokens")
plt.ylabel("Number of samples")
plt.savefig(f"{output}.png")
plt.close()