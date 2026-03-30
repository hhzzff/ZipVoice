if [[ ":$PYTHONPATH:" != *":$(pwd):"* ]]; then
    export PYTHONPATH="$PYTHONPATH:../../."
fi
export CUDA_VISIBLE_DEVICES="0,1"

EXP_DIR="exp/zipvoice_libritts_1121_1109_modified_tokens_durations_rms0"
RES_DIR="${EXP_DIR/exp/res}"
CHECKPOINT_NAME="epoch-60.pt"
EXTRA="-tshift0.7-step8"

python3 -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice \
    --model-dir "$EXP_DIR" \
    --checkpoint-name "$CHECKPOINT_NAME" \
    --tokenizer libritts \
    --test-list test.tsv \
    --res-dir "${RES_DIR}/${CHECKPOINT_NAME}${EXTRA}" \
    --num-step 8 \
    --guidance-scale 1 \
    --t-shift 0.7 \
    --feat-scale 1.0 \
	--AE-path /star-oss/hanzhifeng/mel-vae-proj/mel-ae/exp/train-libritts-251120-1104-no_layernm_downupsample-overlapdownupsample-rms0 \
	--AE-checkpoint 50k