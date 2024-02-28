"""Example Python client for vllm.entrypoints.api_server"""

import argparse
import json
from typing import Iterable, List

import requests


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
    use_beam_search: bool,
    stream: bool,
) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    pload = {
        "prompt": prompt,
        "use_beam_search": use_beam_search,
        "temperature": temperature,
        "max_tokens": max_tokens,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--temperature", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument(
        "--prompt", type=str, default="Gime me ABC notation for some Irish music babe"
    )
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--use_beam_search", action="store_false")

    args = parser.parse_args()
    prompt = args.prompt
    api_url = f"http://{args.host}:{args.port}/generate"
    stream = args.stream

    response = post_http_request(
        prompt, api_url, args.temperature, args.max_tokens, args.use_beam_search, stream
    )

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
