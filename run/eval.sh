python3 -m zipvoice.eval.wer.hubert \
    --wav-path res/zipvoice_libritts_0519_1237_stream_alignmask_fixedwindow_crossattn/epoch-160.pt-tshift0.7-step8-guide1-ratio-noisefix \
    --test-list test.tsv \
    --model-dir download/tts_eval_models \
    --decode-path res/zipvoice_libritts_0519_1237_stream_alignmask_fixedwindow_crossattn/epoch-160.pt-tshift0.7-step8-guide1-ratio-noisefix/wer.txt