if [[ ":$PYTHONPATH:" != *":$(pwd):"* ]]; then
    export PYTHONPATH="$PYTHONPATH:../../."
fi
export CUDA_VISIBLE_DEVICES="0,1,2,3"

TS="$(date +%m%d_%H%M)"
EXP_DIR="exp_pointer/libritts_${TS}"
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
echo "Copied train_word_pointer.sh and $CONFIG_FILE to $EXP_DIR"

PRETRAIN_FLAG=()
if [ -n "$PRETRAINED_CKPT" ]; then
    PRETRAIN_FLAG=(--pretrained-ckpt "$PRETRAINED_CKPT")
    echo "Warm-starting WordPointer (strict=False) from $PRETRAINED_CKPT"
else
    echo "Training WordPointer from scratch."
fi

python3 -m zipvoice.bin.train_word_pointer \
    --manifest-dir aligned_data/fbank \
    --token-file data/tokens_libritts.txt \
    --tokenizer libritts \
    --model-config "$CONFIG_FILE" \
    --exp-dir "$EXP_DIR" \
    --max-duration 250 \
    --feat-scale 0.1 \
    --chunk-frames 150 \
    --max-pad 4 \
    --dim 256 \
    --mel-encoder-layers 4 \
    --text-encoder-layers 4 \
    --cross-attn-layers 4 \
    --num-heads 8 \
    --feedforward-dim 1024 \
    --dropout 0.05 \
    --steps 150000 \
    --lr 3e-4 \
    --augment-prob 0.8 \
    --volume-augment-prob 0.5 \
    --volume-db-min -6 \
    --volume-db-max 6 \
    --speed-augment-prob 0.5 \
    --speed-min 0.9 \
    --speed-max 1.1 \
    --noise-augment-prob 0.3 \
    --noise-std-min 0.005 \
    --noise-std-max 0.03 \
    --eval-every 500 \
    --num-eval-batches 200 \
    --num-tb-samples 6 \
    --tb-sample-topk 5 \
    --seed 42 \
    "${PRETRAIN_FLAG[@]}"
