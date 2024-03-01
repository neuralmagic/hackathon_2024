# hackathon_2024

# Running app locally

### set up vLLM to be accessible locally

ssh tunnel into gpu and run run_server
set gpu port (assuming port 8000) to something local via:
ssh -N -f -L localhost:8000:localhost:8000 [with ssh tunnel address here]

### now, locally, separate from gpu, run the app bump_tunes:

#### set up

pip install 'bump_tunes' requirements.txt
npm i

#### run

uvicorn main:app --port 8080 --reload

in a different window:
npx parcel static/index.html

visit localhost:1234
