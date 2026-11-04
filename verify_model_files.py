#!/usr/bin/env python3
import os, sys
required = ['model.bin', 'config.json', 'source.spm', 'target.spm']
missing = [f for f in required if not os.path.exists(os.path.join('./model', f))]
if missing:
    sys.exit(f'ERROR: faltan archivos en ./model: {missing}')
print('OK: modelo CTranslate2 presente y completo')
for f in sorted(os.listdir('./model')):
    p = os.path.join('./model', f)
    if os.path.isfile(p):
        print(f'  {f}  ({os.path.getsize(p)} bytes)')
