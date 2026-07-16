"""
Download zoogdiergeluiden via NatureLM dataset (Earth Species Project) van Hugging Face.

Herschreven 2026-07-16 — twee bugs gefixt t.o.v. de vorige (streaming) aanpak:

1. TE SMALLE MATCHING: de oude aanpak filterde alleen op `task` bevat
   "taxonomic" en pakte de laatste 2 woorden van het `output`-tekstveld als
   soortnaam. Dat mist minstens 13 andere task-types (genus-detection,
   species-common-detection, family-detection, caption-common, ...) en breekt
   bij multi-species output. Elk record heeft echter ook een `metadata`-veld
   (JSON-string) met een losse, gestructureerde `species`-kolom — ongeacht
   task-type. Een test op 200.000 samples (0,8% van de dataset) liet zien dat
   matchen op metadata.species 8,5x meer treffers oplevert (212 vs 25) over
   2x zoveel soorten (16 vs 8) — inclusief 8 soorten die de oude matching
   volledig miste (o.a. jakhals, wilde kat, steenmarter, zevenslaper).

2. TE LANGZAAM: rij-voor-rij streamen van de dataset (audio + tekst per
   record) duurde voor een steekproef van 0,8% van de dataset bijna een
   volledige dag. DuckDB kan rechtstreeks op de Parquet-bestanden op
   HuggingFace queryen met kolomprojectie — alleen de (lichte) metadata-kolom
   lezen, de (zware) audio-kolom overslaan. Een test op 10 van de 10.577
   Parquet-bestanden duurde 64 seconden, wat een volledige scan van de
   dataset schat op ~19 uur i.p.v. dagen tot weken.
   (Dit idee — DuckDB op de Parquet-bestanden — is al eerder geprobeerd, zie
   git-historie PR #9. Het werkte toen niet: de aanroep naar de HuggingFace
   datasets-server API voor de bestandenlijst had geen Authorization-header,
   en deze dataset is gated — zonder token kwamen er 0 bestanden terug. Met
   een Bearer-token werkt het gewoon.)

Nieuwe strategie, twee fases:
  Fase 1 — INDEX BOUWEN (`build_metadata_index`): scant alle Parquet-
    bestanden met DuckDB, matcht metadata.species tegen species_targets.yaml
    (ongeacht task-type), schrijft treffers (id, species, bronbestand) naar
    een JSONL-index op schijf. Checkpoint per verwerkte batch bestanden —
    hervat na onderbreking vanaf de laatste batch, niet vanaf 0.
  Fase 2 — AUDIO DOWNLOADEN (`download_audio_from_index`): leest de index,
    groepeert per Parquet-bestand en haalt per bestand gericht alleen de
    audio op voor de id's die al matchten (geen audio-kolom-scan van de hele
    dataset). Dedupliceert op `file_name`: dezelfde brondata-opname staat
    vaak onder meerdere task-types in de dataset (bijv. dezelfde clip als
    taxonomic-classification én genus-detection én caption-common) — zonder
    dedupe zou dezelfde audio meermaals gedownload worden.

Vereist: HF_TOKEN env var, of leesbaar via /mnt/usb/hf_cache/token
(de dataset is gated).

Gebruik (index bouwen + audio downloaden, standaard):
    python dataset/download_naturelm.py \
        --output /mnt/usb/prepared \
        --species-file dataset/species_targets.yaml \
        --max-per-species 500

Gebruik (alleen fase 1 — index bouwen/hervatten, geen audio):
    python dataset/download_naturelm.py --index-only

Gebruik (fase 1 overslaan — audio downloaden o.b.v. bestaande index):
    python dataset/download_naturelm.py --skip-index

Gebruik (index helemaal opnieuw opbouwen, checkpoint negeren):
    python dataset/download_naturelm.py --rebuild-index

Gebruik (background downloaden via WavCaps/UrbanSound/AudioCaps — ongewijzigd
t.o.v. voorheen, deze matcht op source_dataset en had de bug hierboven niet):
    python dataset/download_naturelm.py \
        --output /mnt/usb/prepared \
        --skip-species \
        --background-clips 2000
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import urllib.request
import wave
from pathlib import Path

import yaml

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

NATURELM_DATASET = "EarthSpeciesProject/NatureLM-audio-training"
HF_PARQUET_API = (
    "https://datasets-server.huggingface.co/parquet"
    "?dataset=EarthSpeciesProject%2FNatureLM-audio-training"
)
SAMPLE_RATE = 16000
BACKGROUND_SOURCES = {"WavCaps", "UrbanSound", "UrbanSound8K", "AudioCaps"}

INDEX_FILE = Path("/mnt/usb/naturelm_metadata_index.jsonl")
INDEX_CHECKPOINT_FILE = Path("/mnt/usb/naturelm_index_checkpoint.json")
DOWNLOADED_FILENAMES_FILE = Path("/mnt/usb/naturelm_downloaded_filenames.json")


def _slug(scientific: str) -> str:
    return re.sub(r"\s+", "_", scientific.strip().lower())


def _load_species(yaml_path: Path) -> list[dict]:
    with yaml_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data["species"]


def _count_existing(species_dir: Path) -> int:
    """Tel bestaande WAV-bestanden — dit is de voortgang na herstart."""
    if not species_dir.exists():
        return 0
    return sum(1 for _ in species_dir.glob("*.wav"))


def _get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return token
    token_file = Path("/mnt/usb/hf_cache/token")
    if token_file.exists():
        return token_file.read_text().strip()
    print("⚠ Geen HF_TOKEN gevonden (env var of /mnt/usb/hf_cache/token)", file=sys.stderr)
    sys.exit(1)


def _get_parquet_urls(token: str) -> list[str]:
    """Haal Parquet CDN URL's op via de HuggingFace Datasets Server API.

    NatureLM-audio-training is een gated dataset — zonder Authorization-
    header geeft deze API 0 bestanden terug (dit brak de eerdere DuckDB-
    poging in PR #9, die zonder token deed en concludeerde dat "DuckDB niet
    werkt met HF parquet"; het echte probleem was de ontbrekende auth).
    """
    req = urllib.request.Request(
        HF_PARQUET_API,
        headers={"User-Agent": "mammal-watcher/1.0", "Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    urls = [f["url"] for f in data.get("parquet_files", []) if f.get("url")]
    return sorted(urls)


def _duckdb_connection(token: str):
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(f"CREATE SECRET hf_token (TYPE http, BEARER_TOKEN '{token}');")
    return con


def _load_index_checkpoint() -> int:
    try:
        if INDEX_CHECKPOINT_FILE.exists():
            return int(json.loads(INDEX_CHECKPOINT_FILE.read_text()).get("processed_files", 0))
    except Exception:
        pass
    return 0


def _save_index_checkpoint(processed_files: int) -> None:
    try:
        INDEX_CHECKPOINT_FILE.write_text(json.dumps({"processed_files": processed_files}))
    except Exception:
        pass


def _existing_index_ids() -> set[str]:
    """id's die al in de index staan — voorkomt dubbele regels bij hervatten."""
    ids: set[str] = set()
    if INDEX_FILE.exists():
        with INDEX_FILE.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    ids.add(json.loads(line)["id"])
                except Exception:
                    continue
    return ids


def _load_downloaded_filenames() -> dict[str, set[str]]:
    """file_name's die in eerdere download-runs al zijn opgeslagen, per soort.

    Zonder dit vergeet elke nieuwe run van download_audio_from_index welke
    brondata-opnames al gedownload zijn — de dedupe in `_plan_downloads`
    werkt dan alleen BINNEN één run, niet ERTUSSEN. Aangezien de index
    gestaag groeit (checkpointed scan) en de audio-download los daarvan
    meerdere keren gedraaid kan worden, zou dat dezelfde opname onder een
    ander task-type/id opnieuw kunnen downloaden.
    """
    try:
        if DOWNLOADED_FILENAMES_FILE.exists():
            raw = json.loads(DOWNLOADED_FILENAMES_FILE.read_text())
            return {slug: set(names) for slug, names in raw.items()}
    except Exception:
        pass
    return {}


def _save_downloaded_filenames(mapping: dict[str, set[str]]) -> None:
    try:
        DOWNLOADED_FILENAMES_FILE.write_text(
            json.dumps({slug: sorted(names) for slug, names in mapping.items()})
        )
    except Exception:
        pass


def _load_index() -> list[dict]:
    entries: list[dict] = []
    if INDEX_FILE.exists():
        with INDEX_FILE.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    continue
    return entries


def build_metadata_index(
    species_list: list[dict],
    token: str,
    batch_size: int = 200,
    force_rebuild: bool = False,
    debug: bool = False,
) -> None:
    """Fase 1: scan alle Parquet-bestanden op metadata.species, schrijf treffers naar INDEX_FILE."""
    if not HAS_DUCKDB:
        print("⚠ pip install duckdb", file=sys.stderr)
        sys.exit(1)

    if force_rebuild:
        INDEX_FILE.unlink(missing_ok=True)
        INDEX_CHECKPOINT_FILE.unlink(missing_ok=True)

    target_names = {s["scientific"].lower() for s in species_list}
    name_to_slug = {s["scientific"].lower(): _slug(s["scientific"]) for s in species_list}

    print("📡 Parquet-bestandenlijst ophalen (HuggingFace datasets-server, met auth)...")
    urls = _get_parquet_urls(token)
    print(f"  {len(urls)} bestanden gevonden")

    start_at = _load_index_checkpoint()
    if start_at:
        print(f"⏩ Index-checkpoint: hervatten vanaf bestand {start_at}/{len(urls)}")

    con = _duckdb_connection(token)
    seen_ids = _existing_index_ids()
    new_matches = 0
    processed = start_at

    remaining_urls = urls[start_at:]
    batches = [remaining_urls[i:i + batch_size] for i in range(0, len(remaining_urls), batch_size)]
    iterator = tqdm(batches, desc="Index bouwen", unit=" batch") if HAS_TQDM else batches

    with INDEX_FILE.open("a", encoding="utf-8") as out:
        for batch in iterator:
            url_list = ", ".join(f"'{u}'" for u in batch)
            try:
                rows = con.execute(f"""
                    SELECT id, file_name, task, filename AS parquet_url,
                           json_extract_string(metadata, '$.species') AS species
                    FROM read_parquet([{url_list}], union_by_name=true, filename=true)
                    WHERE json_extract_string(metadata, '$.species') IS NOT NULL
                """).fetchall()
            except Exception as exc:
                print(f"\n⚠ Batch overgeslagen ({exc})", file=sys.stderr)
                processed += len(batch)
                _save_index_checkpoint(processed)
                continue

            for row_id, file_name, task, parquet_url, species in rows:
                if not species or species.lower() not in target_names or row_id in seen_ids:
                    continue
                seen_ids.add(row_id)
                slug = name_to_slug[species.lower()]
                out.write(json.dumps({
                    "id": row_id,
                    "species_slug": slug,
                    "file_name": file_name,
                    "parquet_url": parquet_url,
                    "task": task,
                }) + "\n")
                new_matches += 1
                if debug:
                    print(f"  + {slug}: {row_id} ({task})")

            out.flush()
            processed += len(batch)
            _save_index_checkpoint(processed)

    con.close()
    print(f"\n✓ Index compleet: {new_matches} nieuwe treffers toegevoegd (totaal {len(seen_ids)} in index)")


def _save_audio_as_wav(audio_data, dest: Path, sample_rate: int = SAMPLE_RATE) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        # torchcodec backend: AudioDecoder → AudioSamples
        try:
            from torchcodec.decoders import AudioDecoder as _AudioDecoder
            if isinstance(audio_data, _AudioDecoder):
                samples = audio_data.get_all_samples()
                sr = samples.sample_rate or sample_rate
                if HAS_NUMPY and HAS_SOUNDFILE:
                    array = samples.data.numpy()
                    if array.ndim == 2:
                        array = array.mean(axis=0)
                    array = array.astype("float32")
                    sf.write(str(dest), array, sr, subtype="PCM_16")
                    return True
        except ImportError:
            pass

        if isinstance(audio_data, dict):
            # Ruwe Parquet-struct via DuckDB: {'bytes': <flac/wav-encoded>, 'path': ...}
            raw_bytes = audio_data.get("bytes")
            if raw_bytes and HAS_SOUNDFILE:
                array, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
                sf.write(str(dest), array, sr, subtype="PCM_16")
                return True

            # HF datasets Audio-feature (al gedecodeerd): {'array': [...], 'sampling_rate': ...}
            array = audio_data.get("array")
            sr = audio_data.get("sampling_rate", sample_rate)
            if array is not None and HAS_SOUNDFILE:
                if HAS_NUMPY:
                    array = np.array(array, dtype=np.float32)
                sf.write(str(dest), array, sr, subtype="PCM_16")
                return True

        if isinstance(audio_data, (bytes, bytearray)) and HAS_SOUNDFILE:
            buf = io.BytesIO(audio_data)
            array, sr = sf.read(buf, dtype="float32", always_2d=False)
            sf.write(str(dest), array, sr, subtype="PCM_16")
            return True

        import struct
        if isinstance(audio_data, dict):
            raw_samples = audio_data.get("array", [])
            sr = audio_data.get("sampling_rate", sample_rate)
        else:
            raw_samples = list(audio_data) if audio_data else []
            sr = sample_rate
        if not raw_samples:
            return False
        pcm = [max(-32768, min(32767, int(float(s) * 32767))) for s in raw_samples]
        raw = struct.pack(f"<{len(pcm)}h", *pcm)
        with wave.open(str(dest), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(raw)
        return True
    except Exception as exc:
        print(f"\n  ⚠ Audio opslaan mislukt: {exc}", file=sys.stderr)
        return False


def _plan_downloads(
    entries: list[dict],
    target_slugs: set[str],
    counters: dict[str, int],
    max_per_species: int,
    already_downloaded: dict[str, set[str]] | None = None,
) -> dict[str, list[dict]]:
    """Groepeert index-entries per Parquet-bestand, met quota- en dedupe-logica.

    - Slaat entries over voor soorten die al op quota zitten.
    - Dedupliceert op `file_name`: dezelfde brondata-opname staat vaak onder
      meerdere task-types in de index (bijv. dezelfde clip als
      taxonomic-classification én genus-detection én caption-common) — zonder
      dedupe zou dezelfde audio meermaals ingepland worden.
    - `already_downloaded` (optioneel): file_name's die een VORIGE run van
      download_audio_from_index al heeft opgeslagen. Zonder dit werkt de
      dedupe hierboven alleen binnen één run — bij een tweede run (bijv. na
      hervatten, of met een hoger quotum) zou dezelfde opname alsnog opnieuw
      gedownload kunnen worden onder een ander id/task-type.
    """
    seen_file_names: dict[str, set[str]] = {
        slug: set(already_downloaded.get(slug, ())) if already_downloaded else set()
        for slug in target_slugs
    }
    by_file: dict[str, list[dict]] = {}
    for e in entries:
        slug = e.get("species_slug")
        if slug not in target_slugs or counters.get(slug, 0) >= max_per_species:
            continue
        fname = e.get("file_name") or e["id"]
        if fname in seen_file_names[slug]:
            continue
        seen_file_names[slug].add(fname)
        by_file.setdefault(e["parquet_url"], []).append(e)
    return by_file


def download_audio_from_index(
    species_list: list[dict],
    output_dir: Path,
    max_per_species: int,
    token: str,
    debug: bool = False,
) -> dict[str, int]:
    """Fase 2: download audio voor de gematchte rijen uit de index.

    Groepeert per Parquet-bestand en haalt per bestand gericht alleen de
    audio op voor de specifieke id's die al matchten in fase 1 — geen
    audio-kolom-scan van de hele dataset. Dedupliceert op `file_name`
    (zie `_plan_downloads`).
    """
    if not HAS_DUCKDB:
        print("⚠ pip install duckdb", file=sys.stderr)
        sys.exit(1)

    entries = _load_index()
    if not entries:
        print("⚠ Geen index gevonden — draai eerst zonder --skip-index", file=sys.stderr)
        return {}

    target_slugs = {_slug(s["scientific"]) for s in species_list}
    species_nl = {_slug(s["scientific"]): s.get("nl", _slug(s["scientific"])) for s in species_list}

    counters: dict[str, int] = {slug: _count_existing(output_dir / slug) for slug in target_slugs}

    if sum(counters.values()):
        print(f"📂 Voortgang gevonden: {sum(counters.values())} clips al aanwezig")

    already_downloaded = _load_downloaded_filenames()
    by_file = _plan_downloads(entries, target_slugs, counters, max_per_species, already_downloaded)

    if not by_file:
        print("✓ Alle soorten al op quota (of geen matches in index) — niets te downloaden.")
        return counters

    total_needed = sum(len(v) for v in by_file.values())
    print(f"\n📥 Audio downloaden: {total_needed} unieke clips uit {len(by_file)} Parquet-bestand(en)...")

    con = _duckdb_connection(token)
    saved_total = 0
    errors = 0

    iterator = tqdm(by_file.items(), desc="Audio", unit=" bestand") if HAS_TQDM else iter(by_file.items())
    for parquet_url, file_entries in iterator:
        by_id = {e["id"]: e for e in file_entries}
        id_list = ", ".join(f"'{i}'" for i in by_id)
        try:
            rows = con.execute(f"""
                SELECT id, audio FROM read_parquet('{parquet_url}')
                WHERE id IN ({id_list})
            """).fetchall()
        except Exception as exc:
            print(f"\n⚠ Bestand overgeslagen ({exc})", file=sys.stderr)
            errors += len(file_entries)
            continue

        for row_id, audio in rows:
            entry = by_id.get(row_id)
            if entry is None or not audio:
                continue
            slug = entry["species_slug"]
            if counters[slug] >= max_per_species:
                continue

            sample_id = re.sub(r"[^\w\-]", "_", row_id)
            dest = output_dir / slug / f"{sample_id}.wav"
            if dest.exists():
                counters[slug] += 1
                continue

            if _save_audio_as_wav(audio, dest):
                counters[slug] += 1
                saved_total += 1
                already_downloaded.setdefault(slug, set()).add(entry.get("file_name") or row_id)
                if saved_total % 20 == 0:
                    _save_downloaded_filenames(already_downloaded)
                if debug:
                    print(f"  ✓ {species_nl[slug]:20s}: {dest.name} ({counters[slug]}/{max_per_species})")
            else:
                errors += 1

    con.close()
    _save_downloaded_filenames(already_downloaded)

    print(f"\n📊 Resultaat ({saved_total} nieuw opgeslagen, {errors} fouten):")
    for s in species_list:
        slug = _slug(s["scientific"])
        n = counters.get(slug, 0)
        status = "✓" if n >= max_per_species else ("~" if n > 0 else "✗")
        print(f"  {status} {s['nl']:20s} ({s['scientific']}): {n}/{max_per_species}")

    return counters


def _download_background(
    output_dir: Path,
    sources: set[str],
    max_clips: int,
    debug: bool = False,
) -> int:
    """Eén streaming pass voor WavCaps/UrbanSound/AudioCaps als background-klasse.

    Ongewijzigd t.o.v. voorheen — matcht op source_dataset, niet op
    metadata.species, en had de bug uit de moduledocstring niet.
    """
    if not HAS_DATASETS:
        print("⚠ pip install datasets", file=sys.stderr)
        sys.exit(1)

    bg_dir = output_dir / "background"
    bg_dir.mkdir(parents=True, exist_ok=True)

    existing = {f.stem for f in bg_dir.glob("*.wav")}
    remaining = max_clips - len(existing)
    if remaining <= 0:
        print(f"  Background al volledig ({len(existing)} clips aanwezig)")
        return len(existing)

    print(f"\n🌿 Background streamen ({', '.join(sorted(sources))})...")
    print(f"  Doel: {max_clips} clips, al aanwezig: {len(existing)}, nog nodig: {remaining}")

    try:
        ds = load_dataset(NATURELM_DATASET, split="train", streaming=True)
    except Exception as exc:
        print(f"Dataset laden mislukt: {exc}", file=sys.stderr)
        return len(existing)

    ok = 0
    iterator = tqdm(ds, desc="Background", unit=" samples") if HAS_TQDM else ds

    for sample in iterator:
        if ok >= remaining:
            break

        source = sample.get("source_dataset", "")
        if not any(s.lower() in source.lower() for s in sources):
            continue

        sample_id = re.sub(r"[^\w\-]", "_", (sample.get("id") or ""))
        if not sample_id or sample_id in existing:
            continue

        dest = bg_dir / f"bg_{sample_id}.wav"
        if dest.exists():
            existing.add(sample_id)
            continue

        audio = sample.get("audio")
        if not audio:
            continue

        if _save_audio_as_wav(audio, dest):
            existing.add(sample_id)
            ok += 1
            if debug:
                print(f"  [background/{source}] {dest.name}")

    total = len(existing)
    print(f"  Klaar: {ok} nieuw, {total} totaal background clips")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", default="/mnt/usb/prepared")
    parser.add_argument("--max-per-species", type=int, default=50)
    parser.add_argument("--species-file", default="dataset/species_targets.yaml")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="Aantal Parquet-bestanden per DuckDB-query tijdens indexbouw")
    parser.add_argument("--index-only", action="store_true",
                        help="Alleen fase 1 (index bouwen), geen audio downloaden")
    parser.add_argument("--skip-index", action="store_true",
                        help="Sla fase 1 over, download audio o.b.v. bestaande index")
    parser.add_argument("--rebuild-index", action="store_true",
                        help="Negeer bestaande index/checkpoint, scan opnieuw vanaf begin")
    parser.add_argument("--background-clips", type=int, default=0,
                        help="Aantal background clips (0 = uit)")
    parser.add_argument("--background-sources", default="WavCaps,UrbanSound,AudioCaps")
    parser.add_argument("--skip-species", action="store_true",
                        help="Sla soort-download over, doe alleen background")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    species_file = Path(args.species_file)

    if not species_file.exists():
        print(f"Bestand niet gevonden: {species_file}", file=sys.stderr)
        sys.exit(1)
    if not HAS_SOUNDFILE:
        print("⚠ pip install soundfile", file=sys.stderr)
        sys.exit(1)

    species_list = _load_species(species_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"NatureLM downloader — {len(species_list)} soort(en), max {args.max_per_species} per soort")
    print(f"Dataset : {NATURELM_DATASET}")
    print(f"Uitvoer : {output_dir.resolve()}\n")

    if not args.skip_species:
        token = _get_hf_token()
        if not args.skip_index:
            build_metadata_index(
                species_list, token,
                batch_size=args.batch_size,
                force_rebuild=args.rebuild_index,
                debug=args.debug,
            )
        if not args.index_only:
            download_audio_from_index(
                species_list, output_dir, args.max_per_species, token, debug=args.debug,
            )

    if args.background_clips > 0:
        sources = {s.strip() for s in args.background_sources.split(",")}
        _download_background(output_dir, sources, args.background_clips, debug=args.debug)

    print("\n✅ Klaar!")


if __name__ == "__main__":
    main()
