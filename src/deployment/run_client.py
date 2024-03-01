"""Example Python client for vllm.entrypoints.api_server"""

import argparse
import json
from typing import Iterable, List
from string import Template
import re
import requests

prompt_template = Template("Human: ${inst} </s> Assistant: ")

def clear_line(n: int = 1) -> None:
    LINE_UP = "\033[1A"
    LINE_CLEAR = "\x1b[2K"
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)


def post_http_request(
    prompt: str,
    api_url: str,
    temperature: int,
    max_tokens: int,
    top_k: int,
    top_p: int,
    n: int,
    stream: bool
) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    pload = {
        "prompt": prompt_template.safe_substitute({"inst": prompt}),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_k":top_k,
        "top_p": top_p,
        "n": n,
        "stream": stream,
    }
    response = requests.post(api_url, headers=headers, json=pload, stream=True)
    return response


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(
        chunk_size=8192, decode_unicode=False, delimiter=b"\0"
    ):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"]
            yield output


def get_response(response: requests.Response) -> List[str]:
    data = json.loads(response.content)
    output = data["text"]
    return output

def _is_line_abc_notation(line:str):
    abc_pattern = r'(X:\d+\n(?:[^\n]*\n)+)'
    abc_notation = re.findall(abc_pattern, line)
    if not abc_notation:
        print("WARNING: Unable to extract ABC notation from the output")
    else:
        print("---- Extracted ABC notation ----")
        print(abc_notation[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--temperature", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--top_p", type=int, default=1)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument(
        "--prompt", type=str, default="Develop a musical piece using the given chord progression. 'Dm', 'C', 'Dm', 'Dm', 'C', 'Dm', 'C', 'Dm'"
    )
    parser.add_argument("--stream", action="store_true")

    args = parser.parse_args()
    prompt = args.prompt
    api_url = f"http://{args.host}:{args.port}/generate"
    stream = args.stream

    response = post_http_request(prompt=prompt, 
                                 api_url=api_url, 
                                 temperature=args.temperature, 
                                 max_tokens=args.max_tokens, 
                                 top_k=args.top_k, 
                                 top_p=args.top_p, 
                                 n=args.n, 
                                 stream=stream)

    if stream:
        num_printed_lines = 0
        for h in get_streaming_response(response):
            clear_line(num_printed_lines)
            num_printed_lines = 0
            for i, line in enumerate(h):
                num_printed_lines += 1
                print(f"Beam candidate {i}: {line!r}", flush=True)
    else:
        output = get_response(response)
        for i, line in enumerate(output):
            print(f"Beam candidate {i}: {line!r}", flush=True)
            _is_line_abc_notation(line)


