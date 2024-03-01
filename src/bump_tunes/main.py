from typing import Optional
from fastapi import HTTPException
from pydantic import BaseModel
from pydantic import ValidationError
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from chat_musician_request import post_http_request, get_streaming_response, clear_line, get_response, _is_line_abc_notation, extract_abc_notation
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:1234"],  # Adjust this to match your Parcel server URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RequestModel(BaseModel):
    prompt: str
    temperature: Optional[int]
    max_tokens: Optional[int]
    top_k: Optional[int]
    top_p: Optional[int]
    n: Optional[int]

@app.post("/api")
async def read_root(request: RequestModel):
    print(request.prompt)
    print(request)

    response = make_user_request(request)
    return {'music_segment': response['music_segment'], 'is_valid': response['is_valid'] }

# REST API endpoint
@app.get("/api")
async def read_root():
    reponse = make_user_request()
    return {'music_segment': reponse }


# Serve HTML file
app.mount("/", StaticFiles(directory="static", html=True), name="static")

def make_user_request(request_args: RequestModel):
    host = "localhost"
    port = 8000
    default_args = { 'prompt': "Develop a musical piece using the given chord progression. 'Dm', 'C', 'Dm', 'Dm', 'C', 'Dm', 'C', 'Dm'", 'api_url': f"http://{host}:{port}/generate", 'temperature': 0.2, 'max_tokens': 512, 'top_k': 40, 'top_p': .9, 'n': 1, 'stream': False }

    args = {k: request_args.__dict__.get(k, default_args[k]) if request_args.__dict__.get(k, default_args[k]) is not None else default_args[k] for k in default_args}

    response = post_http_request(prompt=args['prompt'],
                                 api_url=args['api_url'],
                                 temperature=args['temperature'],
                                 max_tokens=args['max_tokens'],
                                 top_k=args['top_k'],
                                 top_p=args['top_p'],
                                 n=args['n'],
                                 stream=args['stream'])
    
    if args['stream']:
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
            if _is_line_abc_notation(line):
                abc_notation = extract_abc_notation(line)
                return { 'is_valid': True, 'music_segment': abc_notation[0] }
            else:
                return { 'is_valid': False, 'music_segment': line }