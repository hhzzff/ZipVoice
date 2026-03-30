if [[ ":$PYTHONPATH:" != *":$(pwd):"* ]]; then
    export PYTHONPATH="$PYTHONPATH:../../."
fi
export CUDA_VISIBLE_DEVICES="0,1,2,3"

EXP_DIR="exp/zipvoice_libritts_0326_1653_stream_alignmask_fixedwindow"
CONFIG_FILE="conf/zipvoice_base.json"

if [ -d "$EXP_DIR" ]; then
    echo "Error: Directory '$EXP_DIR' already exists. Aborting to avoid overwriting."
    exit 1
fi

mkdir -p $EXP_DIR
SCRIPT_PATH="$(readlink -f "$0")"
cp "$SCRIPT_PATH" "$EXP_DIR/"
cp "$CONFIG_FILE" "$EXP_DIR/"
echo "Copied train.sh, $CONFIG_FILE to $EXP_DIR"

python3 -m zipvoice.bin.train_zipvoice_stream_fixedwindow \
    --world-size 4 \
	--use-fp16 0 \
	--num-epochs 200 \
	--max-duration 250 \
	--lr-epochs 10 \
	--max-len 20 \
	--valid-by-epoch 1 \
	--model-config "$CONFIG_FILE" \
	--tokenizer libritts \
	--token-file data/tokens_libritts.txt \
	--dataset libritts \
	--exp-dir "$EXP_DIR" \
	--manifest-dir aligned_data/fbank \
	--feat-scale 0.1