#!/usr/bin/env python

import argparse
from pathlib import Path
import sqlite3
import textwrap

DB_EXTS = {".cdb", ".sqlite", ".sqlite3"}

def fline(text, n):
    return "\n".join(textwrap.wrap(text, width=n))

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", type=Path, help="Directorio con las bases SQLite")
    ap.add_argument("--id", required=True, help="ID de la carta")
    ap.add_argument("--width", help="Ancho de línea de la descripción", default=80)
    args = ap.parse_args()

    dbs =  [    x for x in Path(args.input_dir).iterdir()
                if x.is_file() and x.suffix.lower() in DB_EXTS
           ]
    lookup_id = int(args.id)

    for db in dbs:
        con = sqlite3.connect(db)

        tables = sorted(r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'"))
        if 'texts' not in tables:
            print("[!] {} no tiene tabla 'texts'".format(db.name))

        existing = [r[1] for r in con.execute("PRAGMA table_info(texts)")]
        if 'id' not in existing:
            print("[!] {}: Tabla 'texts' no tiene campo 'id'".format(db.name))

        card = con.execute(f"SELECT id, name, desc FROM texts WHERE id = {lookup_id}").fetchone()
        if card is not None:
            id, name, desc = card
            print(f"[ID]: {id}\n"
                  f"[NAME]: {name}\n"
                  f"[DESC]: {fline(desc, int(args.width))}")
            print(f"len(DESC): {len(desc)}")

if __name__ == '__main__':
    main()




