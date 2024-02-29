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
      targets: ['re:model.layers*']
      """

DATASET_NAME="open_platypus"
OUTPUT_DIR="llama7b_oneshot_sparse"
CALIB_SAMPLES=1024
MAX_SEQ_LENGTH=4096
MODEL="m-a-p/ChatMusician"
PAD=False
DEVICE="auto"

# make sure you have installed sparseml-nightly from the source 
one_shot(
        model=MODEL,
        dataset=DATASET_NAME,
        output_dir=OUTPUT_DIR,
        num_calibration_samples=CALIB_SAMPLES,
        recipe=RECIPE,
        max_seq_length = MAX_SEQ_LENGTH,
        pad_to_max_length=PAD,
        oneshot_device=DEVICE)

