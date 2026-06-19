"""
Download zoogdiergeluiden via NatureLM dataset (Earth Species Project) van Hugging Face.

Strategie: één streaming pass — match soort en sla audio direct op.
Voortgang staat op schijf (bestaande WAV-bestanden), geen checkpoint JSON nodig.
Bij herstart worden bestaande clips overgeslagen.

Gebruik (soorten downloaden naar prepared dir):
    python dataset/download_naturelm.py \
        --output /mnt/usb/prepared \
        --species-file dataset/species_targets.yaml \
        --max-per-species 500

Gebruik (background downloaden via WavCaps/UrbanSound/AudioCaps):
    python dataset/download_naturelm.py \
        --output /mnt/usb/prepared \
        --skip-species \
        --background-clips 2000

Gebruik (alles tegelijk):
    python dataset/download_naturelm.py \
        --output /mnt/usb/prepared \
        --max-per-species 500 \
        --background-clips 2000
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
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
SAMPLE_RATE = 16000
BACKGROUND_SOURCES = {"WavCaps", "UrbanSound", "UrbanSound8K", "AudioCaps"}

CHECKPOINT_FILE = Path("/mnt/usb/naturelm_checkpoint.json")


def _load_checkpoint() -> int:
    """Laad streaming positie van vorige afgebroken run."""
    try:
        if CHECKPOINT_FILE.exists():
            return int(json.loads(CHECKPOINT_FILE.read_text()).get("scanned", 0))
    except Exception:
        pass
    return 0


def _save_checkpoint(scanned: int) -> None:
    try:
        CHECKPOINT_FILE.write_text(json.dumps({"scanned": scanned}))
    except Exception:
        pass


def _clear_checkpoint() -> None:
    try:
        CHECKPOINT_FILE.unlink(missing_ok=True)
    except Exception:
        pass



def _slug(scientific: str) -> str:
    return re.sub(r"\s+", "_", scientific.strip().lower())


def _load_species(yaml_path: Path) -> list[dict]:
    with yaml_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data["species"]


def _extract_species_from_output(output: str) -> str:
    """Laatste 2 woorden van taxonomie = genus + soort."""
    if not output or not isinstance(output, str):
        return ""
    words = output.strip().split()
    if len(words) >= 2:
        return " ".join(words[-2:])
    return output.strip()


def _count_existing(species_dir: Path) -> int:
    """Tel bestaande WAV-bestanden — dit is de voortgang na herstart."""
    if not species_dir.exists():
        return 0
    return sum(1 for _ in species_dir.glob("*.wav"))


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


def download_from_naturelm(
    species_list: list[dict],
    output_dir: Path,
    max_per_species: int,
    debug: bool = False,
) -> dict[str, int]:
    """Één streaming pass: match soort en sla audio direct op."""
    if not HAS_DATASETS:
        print("⚠ pip install datasets", file=sys.stderr)
        sys.exit(1)

    # Voortgang van vorige run: tel bestaande WAV-bestanden per soort
    counters: dict[str, int] = {}
    for s in species_list:
        slug = _slug(s["scientific"])
        existing = _count_existing(output_dir / slug)
        counters[slug] = existing

    # Toon startpositie
    total_existing = sum(counters.values())
    if total_existing:
        print(f"📂 Voortgang gevonden: {total_existing} clips al aanwezig")
        for s in species_list:
            slug = _slug(s["scientific"])
            n = counters[slug]
            if n:
                print(f"  ✓ {s['nl']:20s}: {n}/{max_per_species}")

    target_names = {s["scientific"].lower(): _slug(s["scientific"]) for s in species_list}
    species_nl = {_slug(s["scientific"]): s.get("nl", _slug(s["scientific"])) for s in species_list}

    def all_done() -> bool:
        return all(counters[_slug(s["scientific"])] >= max_per_species for s in species_list)

    if all_done():
        print("✓ Alle soorten al op quota — niets te doen.")
        return counters

    needed = {slug: max_per_species - n for slug, n in counters.items() if n < max_per_species}
    print(f"\n📡 Streamen — {sum(needed.values())} clips nodig voor {len(needed)} soort(en)...")
    for s in species_list:
        slug = _slug(s["scientific"])
        if slug in needed:
            print(f"  • {s['nl']:20s}: nog {needed[slug]} clips")

    try:
        ds = load_dataset(NATURELM_DATASET, split="train", streaming=True)
    except Exception as exc:
        print(f"⚠ Dataset laden mislukt: {exc}", file=sys.stderr)
        sys.exit(1)

    start_scanned = _load_checkpoint()
    if start_scanned > 0:
        print(f"\u23e9 Checkpoint: hervatten vanaf sample {start_scanned:,} ...")
        ds = ds.skip(start_scanned)
    scanned = start_scanned

    saved_total = 0
    errors = 0
    iterator = tqdm(ds, desc="Streaming", unit=" samples", initial=start_scanned) if HAS_TQDM else ds

    for sample in iterator:
        if all_done():
            _clear_checkpoint()
            break

        scanned += 1
        if scanned % 10_000 == 0:
            _save_checkpoint(scanned)

        task = sample.get("task") or ""
        if "taxonomic" not in task:
            continue

        output = sample.get("output") or ""
        species_name = _extract_species_from_output(output).lower()

        if species_name not in target_names:
            continue

        slug = target_names[species_name]
        if counters[slug] >= max_per_species:
            continue

        audio = sample.get("audio")
        if not audio:
            continue

        sample_id = re.sub(r"[^\w\-]", "_", sample.get("id") or f"sample_{scanned}")
        dest = output_dir / slug / f"{sample_id}.wav"

        if dest.exists():
            counters[slug] += 1
            continue

        if _save_audio_as_wav(audio, dest):
            counters[slug] += 1
            saved_total += 1
            if debug:
                print(f"  ✓ {species_nl[slug]:20s}: {dest.name} ({counters[slug]}/{max_per_species})")
            elif saved_total % 10 == 0:
                # Periodieke voortgangsupdate zonder debug
                parts = [f"{species_nl[s['scientific'].lower().replace(' ','_')]}: {counters[_slug(s['scientific'])]}" 
                         for s in species_list if counters[_slug(s["scientific"])] > 0]
                print(f"\r  [{saved_total} opgeslagen] " + " | ".join(parts[:5]), end="", flush=True)
        else:
            errors += 1

    else:
        _clear_checkpoint()  # stream volledig doorlopen

    if saved_total:
        print()  # newline na \r updates

    print(f"\n📊 Resultaat ({scanned:,} samples gescand, {saved_total} nieuw opgeslagen, {errors} fouten):")
    for s in species_list:
        slug = _slug(s["scientific"])
        n = counters[slug]
        status = "✓" if n >= max_per_species else ("~" if n > 0 else "✗")
        print(f"  {status} {s['nl']:20s} ({s['scientific']}): {n}/{max_per_species}")

    return counters


def _download_background(
    output_dir: Path,
    sources: set[str],
    max_clips: int,
    debug: bool = False,
) -> int:
    """Eén streaming pass voor WavCaps/UrbanSound/AudioCaps als background-klasse."""
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/mnt/usb/prepared")
    parser.add_argument("--max-per-species", type=int, default=50)
    parser.add_argument("--species-file", default="dataset/species_targets.yaml")
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
        download_from_naturelm(
            species_list,
            output_dir,
            args.max_per_species,
            debug=args.debug,
        )

    if args.background_clips > 0:
        sources = {s.strip() for s in args.background_sources.split(",")}
        _download_background(output_dir, sources, args.background_clips, debug=args.debug)

    print("\n✅ Klaar!")


if __name__ == "__main__":
    main()
