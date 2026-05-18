if [[ ":$PYTHONPATH:" != *":$(pwd):"* ]]; then
    export PYTHONPATH="$PYTHONPATH:../../."
fi
export CUDA_VISIBLE_DEVICES="0,1"

EXP_DIR="exp/zipvoice_libritts_0512_1652_stream_alignmask_fixedwindow_crossattn"
RES_DIR="${EXP_DIR/exp/res}"
CHECKPOINT_NAME="epoch-125.pt"
EXTRA="-tshift0.7-step16-guide1-wp0514"

python3 -m zipvoice.bin.infer_zipvoice_stream_fixed_window_crossattn \
    --model-name zipvoice \
    --model-dir "$EXP_DIR" \
    --checkpoint-name "$CHECKPOINT_NAME" \
    --tokenizer libritts \
    --test-list test.tsv \
    --res-dir "${RES_DIR}/${CHECKPOINT_NAME}${EXTRA}" \
    --num-step 16 \
    --guidance-scale 1 \
    --t-shift 0.7 \
    --feat-scale 0.1 \
    --word-pointer-ckpt exp_pointer/libritts_0514_1704/word_pointer.pt