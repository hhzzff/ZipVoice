if [[ ":$PYTHONPATH:" != *":$(pwd):"* ]]; then
    export PYTHONPATH="$PYTHONPATH:../../."
fi
export CUDA_VISIBLE_DEVICES="4,5,6,7"

EXP_DIR="exp/zipvoice_libritts_0409_1024_stream_alignmask_fixedwindow_crossattn"
CONFIG_FILE="conf/zipvoice_base.json"

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
	--feat-scale 0.1 \
	--start-epoch 4
