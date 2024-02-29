from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:1234"],  # Adjust this to match your Parcel server URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REST API endpoint
@app.get("/api")
async def read_root():
    return {"message": "Hello, this is your API"}

# Serve HTML file
app.mount("/", StaticFiles(directory="static", html=True), name="static")