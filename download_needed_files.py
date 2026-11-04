#!/usr/bin/env python3
from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download

model_id = "Helsinki-NLP/opus-mt-tc-big-en-es"
output_dir = Path("./model")
output_dir.mkdir(parents=True, exist_ok=True)

          # The CTranslate2 converter creates the engine files but does not
          # guarantee that MarianTokenizer's SentencePiece/vocabulary files
          # are present in the output directory.
for filename in ("source.spm", "target.spm", "vocab.json"):
    downloaded = hf_hub_download(repo_id=model_id, filename=filename)
    shutil.copy2(downloaded, output_dir / filename)
