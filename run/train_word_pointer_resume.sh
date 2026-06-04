if [[ ":$PYTHONPATH:" != *":$(pwd):"* ]]; then
    export PYTHONPATH="$PYTHONPATH:../../."
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RESUME_WORD_POINTER_CKPT="${RESUME_WORD_POINTER_CKPT:-exp_pointer/libritts_0514_1704/word_pointer.pt}"
TS="$(date +%m%d_%H%M)"
EXP_DIR="${EXP_DIR:-exp_pointer/libritts_resume_${TS}}"
CONFIG_FILE="conf/zipvoice_base-1500ms.json"

if [ ! -f "$RESUME_WORD_POINTER_CKPT" ]; then
    echo "Error: RESUME_WORD_POINTER_CKPT does not exist: $RESUME_WORD_POINTER_CKPT"
    echo "Set it explicitly, e.g.:"
    echo "  RESUME_WORD_POINTER_CKPT=exp_pointer/xxx/word_pointer.pt bash run/train_word_pointer_resume.sh"
    exit 1
fi

if [ -d "$EXP_DIR" ]; then
    echo "Error: Directory '$EXP_DIR' already exists. Aborting to avoid overwriting."
    exit 1
fi

mkdir -p "$EXP_DIR"
SCRIPT_PATH="$(readlink -f "$0")"
cp "$SCRIPT_PATH" "$EXP_DIR/"
cp "$CONFIG_FILE" "$EXP_DIR/"
echo "Copied train_word_pointer_resume.sh and $CONFIG_FILE to $EXP_DIR"
echo "Resuming WordPointer from $RESUME_WORD_POINTER_CKPT"

python3 -m zipvoice.bin.train_word_pointer \
    --manifest-dir aligned_data/fbank \
    --token-file data/tokens_libritts.txt \
    --tokenizer libritts \
    --model-config "$CONFIG_FILE" \
    --exp-dir "$EXP_DIR" \
    --resume-word-pointer-ckpt "$RESUME_WORD_POINTER_CKPT" \
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
    --steps 250000 \
    --lr 6e-4 \
    --min-lr-ratio 0.2 \
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
    --pause-augment-prob 0.3 \
    --pause-extra-min-frames 8 \
    --pause-extra-max-frames 40 \
    --pause-min-gap-frames 2 \
    --eval-every 500 \
    --num-eval-batches 200 \
    --num-tb-samples 6 \
    --tb-sample-topk 5 \
    --seed 42
