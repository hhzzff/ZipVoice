if [[ ":$PYTHONPATH:" != *":$(pwd):"* ]]; then
    export PYTHONPATH="$PYTHONPATH:../../."
fi
export CUDA_VISIBLE_DEVICES="0"

EXP_DIR="exp/zipvoice_libritts_1202_2345_modified_tokens_durations_vae1_clamp(-4,0)_bad"
RES_DIR="res/GT_libritts_1203_1117_modified_tokens_durations_vae1_dim64"
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
	--VAE-path "/star-oss/hanzhifeng/mel-vae-proj/mel-ae/exp/train-libritts-251205-2209-no_layernm_downupsample-overlapdownupsample-kl1dim64" \
	--VAE-checkpoint best