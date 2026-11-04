#!/usr/bin/env python3
"""
Copia las bases SQLite de un directorio a una ruta de salida, traduce las
columnas name y desc de la tabla `texts` en las COPIAS y las actualiza.
Los originales no se tocan.

Las bases que ya existan en el directorio de destino se OMITEN (así puedes
añadir bases nuevas y solo se traducen las nuevas). Usa --overwrite para rehacerlas.

Usa exclusivamente Marian NMT (Helsinki-NLP) EN->ES con CTranslate2
(offline, sin límites ni claves).

Instalación:
    pip install ctranslate2 transformers sentencepiece

Ejemplos:
    python traducir_texts.py ./dbs -o ./dbs_es --model ./opus_en_es_ct2
    python traducir_texts.py ./dbs -o ./dbs_es --model ./opus_en_es_ct2 --device cuda
    python traducir_texts.py ./dbs -o traducido --model ./modelo --columns name,desc --dry-run
"""
import argparse
import json
import os
import shutil
import socket
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

VERSION = "r1-marian-en-es-ct2-omitir-existentes"
DB_EXTENSIONS = {".cdb", ".db", ".sqlite", ".sqlite3"}

SRC_LANG = "en"
DST_LANG = "es"

ISO_LANG_MAP = {
    "es": "es", "español": "es", "espanol": "es", "spanish": "es",
    "en": "en", "inglés": "en", "ingles": "en", "english": "en",
}


def log(msg=""):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def short(t, n=90):
    t = t.replace("\n", " ⏎ ")
    return t if len(t) <= n else t[: n - 1] + "…"


class Stats:
    def __init__(self):
        self.total = 0
        self.done = 0
        self.errors = 0
        self.inflight = {}
        self.last_ok = time.monotonic()


class MarianTranslator:
    """Traductor Marian NMT (Helsinki-NLP) EN->ES con CTranslate2,
    caché en disco y logs."""

    def __init__(self, model_path, cache_file, stats,
                 device="cpu", compute_type="int8", retries=3):
        self.model_path = model_path
        self.retries = retries
        self.stats = stats
        self._lock = threading.Lock()  # CTranslate2 no es thread-safe en el mismo Translator
        self.cache_file = Path(cache_file)
        self.cache = {}
        if self.cache_file.exists():
            self.cache = json.loads(self.cache_file.read_text(encoding="utf-8"))
            log(f"Caché cargada: {len(self.cache)} textos de {self.cache_file}")
        else:
            log(f"Caché nueva: {self.cache_file}")

        import ctranslate2
        import transformers

        log(f"Cargando modelo Marian EN->ES desde: {model_path}")
        self.tokenizer = transformers.MarianTokenizer.from_pretrained(model_path)
        self.translator = ctranslate2.Translator(
            model_path, device=device, compute_type=compute_type)
        log(f"Modelo cargado (device={device}, compute_type={compute_type})")

    def save(self):
        self.cache_file.write_text(
            json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")

    def _engine_translate(self, text):
        # Línea a línea para conservar saltos y viñetas
        out_lines = []
        for line in text.split("\n"):
            if not line.strip():
                out_lines.append(line)
                continue
            with self._lock:
                tokens = self.tokenizer.convert_ids_to_tokens(
                    self.tokenizer.encode(line))
                # Quitar el ultimo <s> ?
                results = self.translator.translate_batch(
                    [tokens],
                    beam_size=4,
                    max_decoding_length=512,
                )
                translated = self.tokenizer.decode(
                    self.tokenizer.convert_tokens_to_ids(
                        results[0].hypotheses[0]),
                    skip_special_tokens=True)
            out_lines.append(translated)
        return "\n".join(out_lines)

    def probe(self):
        log("Prueba de traducción (Marian EN->ES): traduciendo 'Hello'...")
        t0 = time.monotonic()
        try:
            out = self._engine_translate("Hello")
        except Exception as exc:
            log(f"PRUEBA FALLIDA tras {time.monotonic() - t0:.1f}s: {exc}")
            sys.exit("Revisa el modelo Marian y la configuración de CTranslate2.")
        log(f"Prueba OK ({time.monotonic() - t0:.1f}s): Hello -> {out}")

    def _call(self, text):
        for attempt in range(1, self.retries + 1):
            token = object()
            self.stats.inflight[token] = time.monotonic()
            try:
                out = self._engine_translate(text)
                return out if out is not None else text
            except Exception as exc:
                msg = str(exc).replace("\n", " ")[:200]
                if attempt == self.retries:
                    raise
                secs = 2 ** attempt
                log(f"  ! Fallo con {short(text, 40)!r}: {msg} "
                    f"-> reintento {attempt}/{self.retries - 1} en {secs}s")
                time.sleep(secs)
            finally:
                self.stats.inflight.pop(token, None)
        return text

    def translate(self, text):
        if text in self.cache:
            return self.cache[text]
        result = self._call(text)
        self.cache[text] = result
        return result


def start_heartbeat(stats, interval):
    stop = threading.Event()

    def run():
        while not stop.wait(interval):
            now = time.monotonic()
            ages = [now - t for t in list(stats.inflight.values())]
            oldest = f"{max(ages):.0f}s" if ages else "-"
            log(f"♥ {stats.done}/{stats.total} hechos | peticiones en curso: {len(ages)} "
                f"(la más antigua: {oldest}) | errores: {stats.errors} | "
                f"último éxito hace {now - stats.last_ok:.0f}s")

    threading.Thread(target=run, daemon=True).start()
    return stop


def is_sqlite(path):
    try:
        with open(path, "rb") as f:
            return f.read(16) == b"SQLite format 3\x00"
    except OSError:
        return False


def process_db(src_path, out_dir, tr, stats, columns, workers, dry_run, show, hb):
    log("")
    log(f"=== {src_path.name}")
    if not is_sqlite(src_path):
        log("  - No es una base SQLite (cabecera inválida). Se omite.")
        return

    dst_path = out_dir / src_path.name
    work_path = out_dir / (src_path.name + ".partial")
    if dry_run:
        log("  - --dry-run: se analiza el original en solo lectura (no se copia nada).")
        con = sqlite3.connect(src_path.resolve().as_uri() + "?mode=ro", uri=True)
    else:
        log(f"  - Copiando a {work_path.name} (se renombrará a {dst_path.name} "
            f"al terminar sin errores)")
        shutil.copy2(src_path, work_path)
        con = sqlite3.connect(work_path)

    finalize = False
    updates = []
    try:
        tables = sorted(r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"))
        log(f"  - Tablas encontradas: {', '.join(tables) or '(ninguna)'}")
        if "texts" not in tables:
            log("  - No tiene tabla 'texts'. Se omite (no se crea nada en el destino).")
            return

        existing = [r[1] for r in con.execute("PRAGMA table_info(texts)")]
        log(f"  - Columnas de texts: {', '.join(existing)}")
        use = [c for c in columns if c in existing]
        missing = [c for c in columns if c not in existing]
        if missing:
            log(f"  - Columnas pedidas que NO existen (se ignoran): {', '.join(missing)}")
        if not use:
            log("  - Ninguna columna a traducir. Se omite.")
            return

        cols_sql = ", ".join(f'"{c}"' for c in use)
        rows = con.execute(f"SELECT id, {cols_sql} FROM texts").fetchall()
        ordered = list(dict.fromkeys(
            v for r in rows for v in r[1:] if isinstance(v, str) and v.strip()))
        in_cache = sum(1 for t in ordered if t in tr.cache)
        log(f"  - {len(rows)} filas | columnas a traducir: {', '.join(use)}")
        log(f"  - {len(ordered)} textos únicos: {in_cache} ya en caché, "
            f"{len(ordered) - in_cache} por traducir")

        if dry_run:
            log("  - --dry-run: no se traduce ni se modifica nada.")
            return

        kind = {}
        for r in rows:
            for c, v in zip(use, r[1:]):
                if isinstance(v, str):
                    kind.setdefault(v, c)

        def work(text):
            was_cached = text in tr.cache
            try:
                out = tr.translate(text)
                return (True, text, out, was_cached, None)
            except Exception as exc:
                msg = str(exc).replace("\n", " ")[:200]
                return (False, text, None, False, msg)

        pending_texts = [t for t in ordered if t not in tr.cache]
        units = [t for t in ordered if t in tr.cache] + pending_texts

        stats.total, stats.done, stats.errors = len(ordered), 0, 0
        stats.last_ok = time.monotonic()
        errors = 0
        t_start = time.monotonic()
        log(f"  - Empezando a traducir ({workers} hilos, latido cada {hb}s)...")
        stop_hb = start_heartbeat(stats, hb)
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(work, u) for u in units]

                def results():
                    for fut in as_completed(futures):
                        yield fut.result()

                try:
                    for ok, orig, new, cached, err in results():
                        stats.done += 1
                        if ok:
                            stats.last_ok = time.monotonic()
                            if show:
                                tag = " [caché]" if cached else ""
                                log(f"  [{stats.done}/{stats.total}] ({kind.get(orig, '?')}){tag} "
                                    f"{short(orig)}  ->  {short(new)}")
                                    #f"{(orig)}  ->  {(new)}")
                            elif stats.done % 25 == 0:
                                log(f"  - Progreso: {stats.done}/{stats.total}")
                        else:
                            errors += 1
                            stats.errors = errors
                            log(f"  ! ERROR definitivo con {short(orig, 40)!r}: {err}")
                            if stats.done == errors and errors >= 10:
                                pool.shutdown(wait=False, cancel_futures=True)
                                tr.save()
                                log("Los primeros 10 textos fallaron todos. Se aborta "
                                    "(caché guardada). Revisa el modelo Marian.")
                                sys.exit(1)
                        if stats.done % 100 == 0:
                            tr.save()
                            log(f"  - Caché guardada ({len(tr.cache)} textos)")
                except KeyboardInterrupt:
                    log("Ctrl+C: cancelando pendientes y guardando caché...")
                    pool.shutdown(wait=False, cancel_futures=True)
                    tr.save()
                    sys.exit(130)
        finally:
            stop_hb.set()
        tr.save()
        log(f"  - Traducción terminada en {time.monotonic() - t_start:.0f}s "
            f"({errors} errores). Caché guardada ({len(tr.cache)} textos)")

        for row in rows:
            new_vals = [
                tr.cache.get(v, v) if isinstance(v, str) and v.strip() else v
                for v in row[1:]
            ]
            updates.append((*new_vals, row[0]))

        set_sql = ", ".join(f'"{c}" = ?' for c in use)
        log("  - Actualizando filas en la copia...")
        with con:
            con.executemany(f"UPDATE texts SET {set_sql} WHERE id = ?", updates)

        if errors == 0:
            finalize = True
        else:
            log(f"  ! {errors} textos fallaron: esta base NO se marcará como terminada "
                f"y no se guardará en el destino. Relanza el script: la caché conserva "
                f"lo traducido y solo se reintentan los fallidos.")
    finally:
        con.close()
        if finalize:
            os.replace(work_path, dst_path)
            log(f"  ✓ {len(updates)} filas actualizadas -> {dst_path}")
        elif not dry_run:
            work_path.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", type=Path, help="Directorio con las bases SQLite")
    ap.add_argument("-o", "--output", type=Path, required=True,
                    help="Ruta donde se guardarán las copias traducidas")
    ap.add_argument("--model", required=True,
                    help="Ruta local o ID de Hugging Face del modelo Marian EN->ES "
                         "en formato CTranslate2 (p. ej. ./opus_en_es_ct2)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="Dispositivo para CTranslate2 (por defecto: cpu)")
    ap.add_argument("--compute-type", default="int8",
                    help="Tipo de cómputo de CTranslate2 (por defecto: int8)")
    ap.add_argument("--columns", default="name,desc",
                    help="Columnas de texts a traducir (por defecto: name,desc)")
    ap.add_argument("--workers", type=int, default=1,
                    help="Hilos en paralelo (por defecto: 1; CTranslate2 no es thread-safe)")
    ap.add_argument("--heartbeat", type=int, default=5,
                    help="Cada cuántos segundos mostrar el estado (por defecto: 5)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Rehacer también las bases que ya existen en el destino")
    ap.add_argument("--quiet", action="store_true",
                    help="No mostrar cada texto traducido")
    ap.add_argument("--dry-run", action="store_true",
                    help="Solo analiza (sin copiar, traducir ni modificar nada)")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    socket.setdefaulttimeout(120)

    log(f"traducir_texts.py versión {VERSION}")
    in_dir = args.input_dir.resolve()
    out_dir = args.output.resolve()
    log(f"Directorio de entrada: {in_dir}")
    log(f"Directorio de salida:  {out_dir}")
    if not in_dir.is_dir():
        sys.exit(f"No existe el directorio: {in_dir}")
    if out_dir == in_dir:
        sys.exit("La ruta de salida no puede ser el mismo directorio de entrada.")
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Motor de traducción: Marian NMT (Helsinki-NLP) EN->ES + CTranslate2")
    log(f"Modelo: {args.model} | device: {args.device} | compute_type: {args.compute_type}")
    columns = [c.strip() for c in args.columns.split(",") if c.strip()]
    log(f"Idiomas: {SRC_LANG} -> {DST_LANG} (fijo) | columnas: {', '.join(columns)} | "
        f"hilos: {args.workers}")

    dbs = sorted(p for p in in_dir.iterdir()
                 if p.is_file() and p.suffix.lower() in DB_EXTENSIONS)
    if not dbs:
        sys.exit(f"No se encontraron bases ({', '.join(sorted(DB_EXTENSIONS))}) en {in_dir}")
    log(f"Bases encontradas ({len(dbs)}): {', '.join(p.name for p in dbs)}")

    valid = [p for p in dbs if is_sqlite(p)]
    invalid = [p for p in dbs if p not in valid]
    if invalid:
        log(f"Se IGNORAN {len(invalid)} archivos que no son SQLite válidos: "
            f"{', '.join(p.name for p in invalid)}")
    dbs = valid
    if not dbs:
        sys.exit("No hay ninguna base SQLite válida en el directorio de entrada.")

    if args.overwrite:
        todo, skipped = dbs, []
    else:
        todo = [p for p in dbs if not (out_dir / p.name).exists()]
        skipped = [p for p in dbs if p not in todo]
    if skipped:
        log(f"Se OMITEN {len(skipped)} bases que ya existen en el destino "
            f"(usa --overwrite para rehacerlas): {', '.join(p.name for p in skipped)}")
    if not todo:
        log("No hay bases nuevas que traducir. Nada que hacer.")
        return
    log(f"Bases por traducir ({len(todo)}): {', '.join(p.name for p in todo)}")

    stats = Stats()
    cache_name = f".cache_marian_{SRC_LANG}_{DST_LANG}.json"
    tr = MarianTranslator(args.model, out_dir / cache_name, stats,
                          device=args.device, compute_type=args.compute_type)
    if not args.dry_run:
        tr.probe()
    t0 = time.monotonic()
    try:
        for db in todo:
            process_db(db, out_dir, tr, stats, columns, args.workers,
                       args.dry_run, not args.quiet, args.heartbeat)
    finally:
        if not args.dry_run:
            tr.save()
    log("")
    log(f"LISTO en {time.monotonic() - t0:.0f}s. Resultado en: {out_dir}")


if __name__ == "__main__":
    main()
