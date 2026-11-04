#!/usr/bin/env python

import argparse
import json
import re
import sqlite3
from pathlib import Path

DB_EXTS = {".cdb", ".sqlite", ".sqlite3"}


def build_pattern(glossary):
    mapping = {e["src"]: e["dst"] for e in glossary}
    if not mapping:
        return None, {}
    terms = sorted(mapping, key=len, reverse=True)
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b",
        flags=re.IGNORECASE | re.UNICODE,
    )
    return pattern, mapping


def match_case(src, dst):
    if src.isupper():
        return dst.upper()
    if src[0].isupper():
        return dst[0].upper() + dst[1:]
    return dst


def apply_glossary(text, pattern, mapping):
    if not text or pattern is None:
        return text

    def repl(m):
        return match_case(m.group(0), mapping[m.group(0).lower()])

    return pattern.sub(repl, text)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input_dir", type=Path, help="Directorio con las bases SQLite")
    ap.add_argument("--cards", type=Path, default=Path("cards.json"),
                    help="JSON con [{id, desc}] para UPDATE manual (por defecto: ./cards.json)")
    ap.add_argument("--glossary", type=Path, default=Path("glossary.json"),
                    help="JSON con [{src, dst}] de sustituciones (por defecto: ./glossary.json)")
    ap.add_argument("--no-cards", action="store_true", help="No aplicar cards.json")
    ap.add_argument("--no-glossary", action="store_true", help="No aplicar glossary.json")
    ap.add_argument("--dry-run", action="store_true", help="No escribe nada")
    args = ap.parse_args()

    # --- cargar entradas ---
    cards = []
    if not args.no_cards:
        cards = json.loads(args.cards.read_text(encoding="utf-8"))
        print(f"cards.json: {len(cards)} cartas")

    pattern, mapping = (None, {})
    if not args.no_glossary:
        glossary = json.loads(args.glossary.read_text(encoding="utf-8"))
        pattern, mapping = build_pattern(glossary)
        print(f"glossary.json: {len(mapping)} términos -> {sorted(mapping)}")

    dbs = sorted(p for p in args.input_dir.iterdir()
                 if p.is_file() and p.suffix.lower() in DB_EXTS)
    if not dbs:
        print(f"[!] No hay bases ({', '.join(sorted(DB_EXTS))}) en {args.input_dir}")
        return

    for db in dbs:
        print(f"\n=== {db.name}")
        con = sqlite3.connect(db)
        try:
            tables = {r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            if "texts" not in tables:
                print(f"[!] {db.name} no tiene tabla 'texts'")
                continue

            existing = [r[1] for r in con.execute("PRAGMA table_info(texts)")]
            if "id" not in existing:
                print(f"[!] {db.name}: Tabla 'texts' no tiene campo 'id'")
                continue
            if "desc" not in existing:
                print(f"[!] {db.name}: Tabla 'texts' no tiene campo 'desc'")
                continue

            # --- FASE 1: cards.json -> UPDATE desc por id ---
            if cards and not args.no_cards:
                updates_cards = []
                for card in cards:
                    card_id = card["id"]
                    card_desc = card["desc"]
                    row = con.execute(
                        "SELECT name FROM texts WHERE id = ?", (card_id,)
                    ).fetchone()
                    if row is None:
                        print(f"[SKIP] id={card_id} no existe en {db.name}")
                        continue
                    updates_cards.append((card_desc, card_id))
                    print(f"[OK] id={card_id} ({row[0]}) -> desc cards.json "
                          f"({len(card_desc)} chars)")
                if updates_cards and not args.dry_run:
                    with con:
                        con.executemany(
                            'UPDATE texts SET "desc" = ? WHERE id = ?',
                            updates_cards,
                        )
                    print(f"[OK] {len(updates_cards)} filas actualizadas desde cards.json")

            # --- FASE 2: glossary.json -> sustituciones sobre desc ---
            if pattern is not None:
                rows = con.execute("SELECT id, \"desc\" FROM texts").fetchall()
                updates_glos = []
                for row_id, desc in rows:
                    if not isinstance(desc, str):
                        continue
                    new_desc = apply_glossary(desc, pattern, mapping)
                    if new_desc != desc:
                        updates_glos.append((new_desc, row_id))

                if updates_glos:
                    if args.dry_run:
                        print(f"[DRY] glosario cambiaría {len(updates_glos)} filas")
                        for new_desc, row_id in updates_glos[:5]:
                            print(f"  ~ id={row_id}: {new_desc[:80]!r}")
                        if len(updates_glos) > 5:
                            print(f"  ... y {len(updates_glos) - 5} más")
                    else:
                        with con:
                            con.executemany(
                                'UPDATE texts SET "desc" = ? WHERE id = ?',
                                updates_glos,
                            )
                        print(f"[OK] glosario aplicado a {len(updates_glos)}/{len(rows)} filas")
                else:
                    print(f"[OK] glosario: ninguna fila cambia ({len(rows)} revisadas)")
        finally:
            con.close()


if __name__ == "__main__":
    main()
