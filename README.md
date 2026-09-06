# Streaming ZipVoice paper demo

This static page presents representative prompt, text, and generated-audio rows
in the style commonly used by TTS paper demo sites. The samples come from the
fixed `test_small100.tsv` evaluation and the current best streaming checkpoint.

```bash
DEMO_PORT=7860 bash run/demo_streaming.sh
```

Open `http://localhost:7860/demo/streaming_tts/`. The server is static and does
not use a GPU.

The packaged demo can also be viewed offline: extract the archive and open
`streaming_tts_demo/index.html` directly in a browser.
