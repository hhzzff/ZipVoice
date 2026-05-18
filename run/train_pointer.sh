if [[ ":$PYTHONPATH:" != *":$(pwd):"* ]]; then
    export PYTHONPATH="$PYTHONPATH:../../."
fi
export CUDA_VISIBLE_DEVICES="0,1,2,3"

TS="$(date +%m%d_%H%M)"
EXP_DIR="exp_pointer/libritts_${TS}"
CONFIG_FILE="conf/zipvoice_base-1500ms.json"
PRETRAINED_CKPT="/star-oss/hanzhifeng/streaming/ZipVoice/egs/zipvoice/exp/zipvoice_libritts_0427_1717_stream_alignmask_fixedwindow_crossattn/checkpoint-140000.pt"

if [ -d "$EXP_DIR" ]; then
    echo "Error: Directory '$EXP_DIR' already exists. Aborting to avoid overwriting."
    exit 1
fi

mkdir -p "$EXP_DIR"
SCRIPT_PATH="$(readlink -f "$0")"
cp "$SCRIPT_PATH" "$EXP_DIR/"
cp "$CONFIG_FILE" "$EXP_DIR/"
echo "Copied train_pointer.sh and $CONFIG_FILE to $EXP_DIR"

PRETRAIN_FLAG=()
if [ -n "$PRETRAINED_CKPT" ]; then
    PRETRAIN_FLAG=(--pretrained-ckpt "$PRETRAINED_CKPT")
    echo "Using pretrained text branch from $PRETRAINED_CKPT"
else
    echo "No PRETRAINED_CKPT set; text branch will train from scratch."
fi

python3 -m zipvoice.bin.train_window_pointer \
    --manifest-dir aligned_data/fbank \
    --token-file data/tokens_libritts.txt \
    --tokenizer libritts \
    --model-config "$CONFIG_FILE" \
    --exp-dir "$EXP_DIR" \
    --max-duration 250 \
    --feat-scale 0.1 \
    --window-size 30 \
    --steps 5000 \
    --lr 5e-4 \
    --eval-every 500 \
    --num-plots 5 \
    --seed 42 \
    "${PRETRAIN_FLAG[@]}"
