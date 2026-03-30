if [[ ":$PYTHONPATH:" != *":$(pwd):"* ]]; then
    export PYTHONPATH="$PYTHONPATH:../../."
fi
export CUDA_VISIBLE_DEVICES="0,1"

EXP_DIR="exp/zipvoice_libritts_0326_1653_stream_alignmask_fixedwindow"
RES_DIR="${EXP_DIR/exp/res}"
CHECKPOINT_NAME="epoch-100.pt"
EXTRA="-tshift0.7-step8"

python3 -m zipvoice.bin.infer_zipvoice_stream_fixed_window \
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