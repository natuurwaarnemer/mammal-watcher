# Dataset Pipeline — Eigen soortherkenningsmodel

Deze map bevat de volledige pipeline voor het downloaden en voorbereiden van trainingsdata voor het eigen NL-zoogdier soortherkenningsmodel (stap 4 in de [ROADMAP](../ROADMAP.md)).

## Overzicht

```
dataset/
├── species_targets.yaml       # Doelsoorten met metadata
├── download_naturelm.py       # Primaire downloader: NatureLM dataset (Hugging Face)
├── download_freesound.py      # Aanvullende downloader: Freesound.org API
├── prepare_dataset.py         # Converteert audio → 16kHz mono WAV chunks
├── extract_features.py        # Extraheert YAMNet 512-dim embeddings
└── train_classifier.py        # Traint SVM/MLP classifier op embeddings

dataset/raw/                   # Ruwe WAV downloads (niet in git)
dataset/prepared/              # Genormaliseerde WAV chunks (niet in git)
dataset/features/              # NumPy feature arrays (niet in git)
models/                        # Getrainde modellen (niet in git)
```

## Stap-voor-stap

### 1. Trainingsdata downloaden (NatureLM — primaire bron)

```bash
python dataset/download_naturelm.py --output dataset/raw --species-file dataset/species_targets.yaml
```

Streamt audio van de [NatureLM dataset](https://huggingface.co/datasets/davidrrobinson/NatureLM-audio)
(Earth Species Project) via Hugging Face. Filtert automatisch op de doelsoorten uit
`species_targets.yaml` op basis van wetenschappelijke naam.

Downloads worden opgeslagen als `dataset/raw/{soort_slug}/*.wav` met metadata in
`dataset/raw/{soort_slug}/metadata.jsonl`.

Opties:
```bash
python dataset/download_naturelm.py --max-per-species 100   # meer opnames per soort
python dataset/download_naturelm.py --dataset andere/dataset # alternatieve HF dataset
```

### 2. Extra trainingsdata (Freesound — optioneel)

```bash
export FREESOUND_API_KEY=<jouw_key>
python dataset/download_freesound.py --output dataset/raw --species-file dataset/species_targets.yaml
```

Vereist een gratis Freesound API key: <https://freesound.org/apiv2/apply/>

Slaat op in hetzelfde formaat als `download_naturelm.py`, zodat `prepare_dataset.py`
beide bronnen samen verwerkt.

### 3. Audio voorbereiden

```bash
python dataset/prepare_dataset.py --input dataset/raw --output dataset/prepared
```

Converteert audio naar 16kHz mono WAV, gesplitst in 5-seconden chunks.
Genereert `dataset/prepared/index.csv` met alle bestanden.

### 4. YAMNet features extraheren

```bash
python dataset/extract_features.py --input dataset/prepared --output dataset/features
```

Laadt YAMNet en extraheert 512-dimensionale embeddings per chunk.
Slaat op als `dataset/features/{soort_slug}.npy` + `labels.npy` + `species_map.json`.

### 5. Classifier trainen

```bash
python dataset/train_classifier.py --features dataset/features --output models/
```

Traint een SVM (RBF kernel) op de YAMNet embeddings.
Slaat op als `models/mammal_classifier_v1.pkl` + `models/species_map.json`.

## Vereisten

```bash
pip install -r requirements.txt
```

Benodigde pakketten:
- `datasets>=2.14` — HuggingFace dataset streaming
- `huggingface-hub>=0.20` — HuggingFace verbinding
- `tqdm` — voortgangsbalken
- `scikit-learn` — SVM/MLP classifier
- `scipy` + `soundfile` — audio I/O
- `tensorflow-hub` — YAMNet model

## Bronnen

| Bron | Inhoud | URL |
|---|---|---|
| NatureLM / Earth Species Project | Zoogdiergeluiden ML-dataset | https://huggingface.co/datasets/davidrrobinson/NatureLM-audio |
| Freesound.org | Open audio community (API key vereist) | https://freesound.org |
| iNaturalist | Observaties met geluid | https://www.inaturalist.org |
| Eigen veldopnames | Active learning | `clips/` map |
