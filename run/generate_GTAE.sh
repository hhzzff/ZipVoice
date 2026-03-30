if [[ ":$PYTHONPATH:" != *":$(pwd):"* ]]; then
    export PYTHONPATH="$PYTHONPATH:../../."
fi
export CUDA_VISIBLE_DEVICES="3"

EXP_DIR="exp/zipvoice_libritts_1125_1057_modified_tokens_durations"
RES_DIR="res/GT_libritts_1204_1548_modified_tokens_durations-normalnoise(0.75,1)-encodebias0"
CHECKPOINT_NAME="epoch-15.pt"
EXTRA="-tshift0.7-step8"

python3 -m zipvoice.bin.infer_zipvoice_identity \
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
	--AE-path "/star-oss/hanzhifeng/mel-vae-proj/mel-ae/exp/train-libritts-251127-1739-no_layernm_downupsample-overlapdownupsample-normalnoise(0.75,1)-encodebias0" \
	--AE-checkpoint best