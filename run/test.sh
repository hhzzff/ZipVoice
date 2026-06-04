if [[ ":$PYTHONPATH:" != *":$(pwd):"* ]]; then
    export PYTHONPATH="$PYTHONPATH:../../."
fi
export CUDA_VISIBLE_DEVICES="0,1"

EXP_DIR="exp/zipvoice_libritts_0519_1237_stream_alignmask_fixedwindow_crossattn"
RES_DIR="${EXP_DIR/exp/res}"
CHECKPOINT_NAME="epoch-189.pt"
EXTRA="-tshift0.7-step8-guide1-wp0514-___-right-1-noisefix-test"

SCRIPT_PATH="$(readlink -f "$0")"
cp "$SCRIPT_PATH" "$RES_DIR/"

python3 -m zipvoice.bin.infer_zipvoice_stream_fixed_window_crossattn \
    --model-name zipvoice \
    --model-dir "$EXP_DIR" \
    --checkpoint-name "$CHECKPOINT_NAME" \
    --tokenizer libritts \
    --test-list test.tsv \
    --res-dir "${RES_DIR}/${CHECKPOINT_NAME}${EXTRA}" \
    --num-step 8 \
    --guidance-scale 1 \
    --t-shift 0.7 \
    --feat-scale 0.1 \
    --word-pointer-ckpt exp_pointer/libritts_0514_1704/word_pointer.pt \
    --advance-mode ratio
    # --save-chunk-wavs 1 \
    # --debug-plot-dir ./debug_plot \
    # --debug-plot-every 1