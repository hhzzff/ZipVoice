if [[ ":$PYTHONPATH:" != *":$(pwd):"* ]]; then
    export PYTHONPATH="$PYTHONPATH:../../."
fi
export CUDA_VISIBLE_DEVICES="0"

TS="$(date +%m%d_%H%M)"
EXP_DIR="exp_ctc/libritts_${TS}"
CONFIG_FILE="conf/zipvoice_base-1500ms.json"
PRETRAINED_CKPT=""

if [ -d "$EXP_DIR" ]; then
    echo "Error: Directory '$EXP_DIR' already exists. Aborting to avoid overwriting."
    exit 1
fi

mkdir -p "$EXP_DIR"
SCRIPT_PATH="$(readlink -f "$0")"
cp "$SCRIPT_PATH" "$EXP_DIR/"
cp "$CONFIG_FILE" "$EXP_DIR/"
echo "Copied train_ctc_aligner.sh and $CONFIG_FILE to $EXP_DIR"

PRETRAIN_FLAG=()
if [ -n "$PRETRAINED_CKPT" ]; then
    PRETRAIN_FLAG=(--pretrained-ckpt "$PRETRAINED_CKPT")
    echo "Warm-starting CTC encoder (strict=False) from $PRETRAINED_CKPT"
else
    echo "Training CTC aligner from scratch."
fi

python3 -m zipvoice.bin.train_ctc_aligner \
    --manifest-dir aligned_data/fbank \
    --token-file data/tokens_libritts.txt \
    --tokenizer libritts \
    --model-config "$CONFIG_FILE" \
    --exp-dir "$EXP_DIR" \
    --max-duration 250 \
    --feat-scale 0.1 \
    --window-size 30 \
    --encoder-layers 6 \
    --encoder-dim 192 \
    --steps 10000 \
    --lr 5e-4 \
    --eval-every 500 \
    --num-eval-batches 50 \
    --num-plots 5 \
    --seed 42 \
    --phase2-start 5000 \
    --phase2-ctc-weight 0.5 \
    --phase2-ce-weight 0.5 \
    "${PRETRAIN_FLAG[@]}"
