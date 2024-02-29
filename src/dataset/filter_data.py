from datasets import load_dataset, concatenate_datasets
from copy import deepcopy

dataset = "m-a-p/MusicPile"
split = "train"

data = load_dataset(dataset, split=split)

def filter_function(example, keys):
    to_filter = False
    for k in keys:
        if k in example["src"].lower():
            to_filter = True
            break

    return to_filter

def parse_function(example):
    text = example["text"]
    index_assistant = text.find("Assistant: ")
    prompt = text[:index_assistant + 11]
    output = text[index_assistant + 11:]
    index_eos = output.find("</s>")
    output = output[:index_eos+4]

    return {"prompt": prompt, "text": output}

irishman_data = data.filter(lambda x: filter_function(x, ["sander-wood/irishman"]))
irishman_data = irishman_data.train_test_split(test_size=2000, train_size=100000)

jsb_data = data.filter(lambda x: filter_function(x, ["sander-wood/deepchoir"]))
jsb_data = jsb_data.train_test_split(test_size=1)

kern_data = data.filter(lambda x: filter_function(x, ["kern"]))
kern_data = kern_data.train_test_split(test_size=2000)

train_data = concatenate_datasets([irishman_data["train"], jsb_data["train"], kern_data["train"]]).shuffle()
train_data = train_data.map(lambda x: parse_function(x))
print(train_data)
train_data.to_json("train.json")

test_data = concatenate_datasets([irishman_data["test"], jsb_data["test"], kern_data["test"]])
test_data = test_data.map(lambda x: parse_function(x))
print(test_data)
test_data.to_json("test.json")