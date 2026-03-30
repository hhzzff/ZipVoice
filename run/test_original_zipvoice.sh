if [[ ":$PYTHONPATH:" != *":$(pwd):"* ]]; then
    export PYTHONPATH="$PYTHONPATH:../../."
fi
export CUDA_VISIBLE_DEVICES="0,1"

EXP_DIR="exp/zipvoice_libritts_0207_1844_stream_alignmask"
RES_DIR="${EXP_DIR/exp/res}"
CHECKPOINT_NAME="epoch-60.pt"
EXTRA="-tshift0.7-step8"

python3 -m zipvoice.bin.infer_zipvoice_partial \
    --model-name zipvoice \
    --model-dir "$EXP_DIR" \
    --checkpoint-name "$CHECKPOINT_NAME" \
    --tokenizer libritts \
    --test-list test.tsv \
    --res-dir "${RES_DIR}/${CHECKPOINT_NAME}${EXTRA}" \
    --num-step 8 \
    --guidance-scale 1 \
    --t-shift 0.7 \
    --feat-scale 0.1