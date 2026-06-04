if [[ ":$PYTHONPATH:" != *":$(pwd):"* ]]; then
    export PYTHONPATH="$PYTHONPATH:../../."
fi
export CUDA_VISIBLE_DEVICES="0,1"

EXP_DIR="exp/zipvoice_libritts_0519_1237_stream_alignmask_fixedwindow_crossattn"
CONFIG_FILE="conf/zipvoice_base-1500ms.json"

python3 -m zipvoice.bin.train_zipvoice_stream_fixedwindow_crossattn \
    --world-size 2 \
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
	--master-port 11451 \
	--start-epoch 75