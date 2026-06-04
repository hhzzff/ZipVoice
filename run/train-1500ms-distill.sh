if [[ ":$PYTHONPATH:" != *":$(pwd):"* ]]; then
    export PYTHONPATH="$PYTHONPATH:../../."
fi
export CUDA_VISIBLE_DEVICES="4,5,6,7"

TS="$(date +%m%d_%H%M)"
EXP_DIR="exp/zipvoice_libritts_${TS}_stream_alignmask_fixedwindow_crossattn_distill"
STUDENT_MODEL="exp/zipvoice_libritts_0512_1652_stream_alignmask_fixedwindow_crossattn/epoch-125.pt"
TEACHER_MODEL="/star-home/hanzhifeng/ZipVoice/egs/zipvoice/exp/zipvoice_libritts/epoch-60-avg-10.pt"
CONFIG_FILE="conf/zipvoice_base-1500ms.json"

if [ -d "$EXP_DIR" ]; then
    echo "Error: Directory '$EXP_DIR' already exists. Aborting to avoid overwriting."
    exit 1
fi

mkdir -p $EXP_DIR
SCRIPT_PATH="$(readlink -f "$0")"
cp "$SCRIPT_PATH" "$EXP_DIR/"
cp "$CONFIG_FILE" "$EXP_DIR/"
echo "Copied train.sh, $CONFIG_FILE to $EXP_DIR"

python3 -m zipvoice.bin.train_zipvoice_stream_fixedwindow_crossattn_distill \
    --world-size 4 \
	--use-fp16 0 \
    --model-config "$CONFIG_FILE" \
    --tokenizer libritts \
    --token-file data/tokens_libritts.txt \
    --dataset libritts \
	--manifest-dir aligned_data/fbank \
    --checkpoint "$STUDENT_MODEL" \
    --teacher-model "$TEACHER_MODEL" \
    --distill-weight 0.1 \
	--distill-loss huber \
	--distill-huber-delta 1.0 \
	--distill-loss-ceil 10.0 \
    --exp-dir "$EXP_DIR"