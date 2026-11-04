#!/usr/bin/env bash
set -eu

python3 -m pip install --upgrade pip
python3 -m pip install "ctranslate2>=3.0" transformers sentencepiece huggingface_hub

command -v ct2-transformers-converter >/dev/null || {
    echo "ERROR: ct2-transformers-converter no está en PATH" >&2
    exit 1
}

mkdir -p model

if [ -f ./model/model.bin ]; then
    echo "[=] Modelo ya convertido en ./model, se omite la conversión."
else
    ct2-transformers-converter \
      --model Helsinki-NLP/opus-mt-tc-big-en-es \
      --output_dir ./model \
      --quantization int8 --force
fi

[ -f ./model/model.bin ] || { echo "ERROR: no se generó model.bin" >&2; exit 1; }

python3 download_needed_files.py
python3 verify_model_files.py

echo "[+] Hecho."
echo "    Ejemplo: python3 marian.py ./en -o ./es --model ./model --device cpu --compute-type int8"
