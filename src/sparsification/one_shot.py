from sparseml.transformers.text_generation import one_shot

RECIPE = """
test_stage:
  obcq_modifiers:
    SparseGPTModifier:
      sparsity: 0.5
      sparsity_profile: owl
      owl_m: 5
      owl_lmbda: 0.08
      block_size: 128
      sequential_update: False
      quantize: False
      targets: 
        - model.layers.0
        - model.layers.1
        - model.layers.2
        - model.layers.3
        - model.layers.4
        - model.layers.5
        - model.layers.6
        - model.layers.7
        - model.layers.8
        - model.layers.9
        - model.layers.10
        - model.layers.11
        - model.layers.12
        - model.layers.13
        - model.layers.14
        - model.layers.15
        - model.layers.16
        - model.layers.17
        - model.layers.18
        - model.layers.19
        - model.layers.20
        - model.layers.21
        - model.layers.22
        - model.layers.23
        - model.layers.24
        - model.layers.25
        - model.layers.26
        - model.layers.27
        - model.layers.28
        - model.layers.29
        - model.layers.30
        - model.layers.31
      """

def custom_mapping(data):
    return {"text": data["prompt"] + data["text"]}

DATASET_NAME = "json"
DATASET_PATH = "/network/damian/hackathon/dataset"
OUTPUT_DIR = "chatmusician_pruned50_oneshot"
CALIB_SAMPLES = 2048
MAX_SEQ_LENGTH = 512
MODEL = "m-a-p/ChatMusician"
PAD = False
DEVICE = "auto"

# make sure you have installed sparseml-nightly from the source 
one_shot(
        model=MODEL,
        dataset=DATASET_NAME,
        dataset_path=DATASET_PATH,
        splits="train",
        remove_columns=["id", "src"],
        output_dir=OUTPUT_DIR,
        preprocessing_func=custom_mapping,
        num_calibration_samples=CALIB_SAMPLES,
        recipe=RECIPE,
        max_seq_length=MAX_SEQ_LENGTH,
        pad_to_max_length=PAD,
        oneshot_device=DEVICE)

