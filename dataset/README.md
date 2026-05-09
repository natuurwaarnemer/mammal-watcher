# Dataset Pipeline — Eigen soortherkenningsmodel

Deze map bevat de volledige pipeline voor het downloaden en voorbereiden van trainingsdata voor het eigen NL-zoogdier soortherkenningsmodel (stap 4 in de [ROADMAP](../ROADMAP.md)).

## Overzicht

```
dataset/
├── species_targets.yaml       # Doelsoorten met metadata en API-queries
├── download_xeno_canto.py     # Downloadt audio van Xeno-Canto API v2
├── prepare_dataset.py         # Converteert MP3 → 16kHz mono WAV chunks
├── extract_features.py        # Extraheert YAMNet 512-dim embeddings
└── train_classifier.py        # Traint SVM/MLP classifier op embeddings

dataset/raw/                   # Ruwe MP3 downloads (niet in git)
dataset/prepared/              # Genormaliseerde WAV chunks (niet in git)
dataset/features/              # NumPy feature arrays (niet in git)
models/                        # Getrainde modellen (niet in git)
```

## Stap-voor-stap

### 1. Trainingsdata downloaden (Xeno-Canto)

```bash
python dataset/download_xeno_canto.py --output dataset/raw --max-per-species 20
```

Downloads worden opgeslagen als `dataset/raw/{soort_slug}/*.mp3` met metadata in
`dataset/raw/{soort_slug}/metadata.jsonl`.

### 2. Audio voorbereiden

```bash
python dataset/prepare_dataset.py --input dataset/raw --output dataset/prepared
```

Converteert MP3 naar 16kHz mono WAV, gesplitst in 5-seconden chunks.
Genereert `dataset/prepared/index.csv` met alle bestanden.

### 3. YAMNet features extraheren

```bash
python dataset/extract_features.py --input dataset/prepared --output dataset/features
```

Laadt YAMNet en extraheert 512-dimensionale embeddings per chunk.
Slaat op als `dataset/features/{soort_slug}.npy` + `labels.npy` + `species_map.json`.

### 4. Classifier trainen

```bash
python dataset/train_classifier.py --features dataset/features --output models/
```

Traint een SVM (RBF kernel) op de YAMNet embeddings.
Slaat op als `models/mammal_classifier_v1.pkl` + `models/species_map.json`.

## Vereisten

```bash
pip install -r requirements.txt
```

Benodigde extra pakketten (zie ook `requirements.txt`):
- `tqdm` — voortgangsbalken
- `scikit-learn` — SVM/MLP classifier
- `scipy` + `soundfile` — audio I/O
- `tensorflow-hub` — YAMNet model

## Bronnen

| Bron | Inhoud | URL |
|---|---|---|
| Xeno-Canto | Vogels + zoogdieren | https://xeno-canto.org |
| Macaulay Library | Cornell Lab, zoogdiergeluiden | https://www.macaulaylibrary.org |
| iNaturalist | Observaties met geluid | https://www.inaturalist.org |
| Berlin Sound Archive | Europese zoogdiergeluiden | https://tierstimmenarchiv.de |
| Eigen veldopnames | Active learning | `clips/` map |
