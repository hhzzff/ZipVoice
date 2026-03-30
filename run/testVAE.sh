if [[ ":$PYTHONPATH:" != *":$(pwd):"* ]]; then
    export PYTHONPATH="$PYTHONPATH:../../."
fi
export CUDA_VISIBLE_DEVICES="6"

EXP_DIR="exp/zipvoice_libritts_1221_0338_modified_tokens_durations_vae1_dim64_str1_weightedfm0.25"
RES_DIR="${EXP_DIR/exp/res}"
CHECKPOINT_NAME="epoch-35.pt"
NUM_STEPS=8
T_SHIFT=0.7
GUIDANCE_SCALE=1.0
EXTRA="-tshift${T_SHIFT}-step${NUM_STEPS}-guidance${GUIDANCE_SCALE}-wot0.5"

python3 -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice \
    --model-dir "$EXP_DIR" \
    --checkpoint-name "$CHECKPOINT_NAME" \
    --tokenizer libritts \
    --test-list test.tsv \
    --res-dir "${RES_DIR}/${CHECKPOINT_NAME}${EXTRA}" \
    --num-step $NUM_STEPS \
    --guidance-scale $GUIDANCE_SCALE \
    --t-shift $T_SHIFT \
    --feat-scale 1.0 \
	--VAE-path "/star-oss/hanzhifeng/mel-vae-proj/mel-ae/exp/train-libritts-251217-1240-no_layernm_downupsample-overlapdownupsample-kl1dim64stride1" \
	--VAE-checkpoint 100k