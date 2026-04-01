# Vietnamese Two-Speaker Separator

A project for building a data pipeline that mixes speech from 2 speakers using the VIVOS dataset, and fine-tuning Conv-TasNet 5M for the Vietnamese speech separation task.
If you're curious, feel free to check out the detailed report of our team by clicking the PDF file.

## Pipeline Overview

1. Audio quality check (filter out files that are too short or near-silent).
2. Pair two different speakers with similar audio lengths.
3. Generate a mixed dataset (mix, s1, s2) at 8 kHz.
4. Fine-tune Conv-TasNet on the generated dataset.

## Environment Requirements

- Python 3.10.13
- Key libraries: `torch 2.7.1+cu118`, `torchaudio 2.7.1+cu118`, `pyloudnorm 0.2.0`, `tqdm`, `wandb`

Quick install:

```bash
pip install torch==2.7.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install pyloudnorm==0.2.0 tqdm wandb
```

## Data

### `data/raw`

Download the dataset from: [https://www.kaggle.com/datasets/kynthesis/vivos-vietnamese-speech-corpus-for-asr](https://www.kaggle.com/datasets/kynthesis/vivos-vietnamese-speech-corpus-for-asr)

* The original VIVOS dataset should be placed in `data/raw/vivos`.

---

### `data/datasets`

This directory contains the mixed dataset generated from the VIVOS corpus. You have two options:

* Download the preprocessed dataset from: [https://www.kaggle.com/datasets/ilewanducki/vivos-mix](https://www.kaggle.com/datasets/ilewanducki/vivos-mix)
* Or run the pipeline script below to automatically generate the mixed dataset from the raw data

## Running the Pipeline

### 1. Audio Quality Check

```bash
python -m src.analysis.quality_check
```

Results are saved to `logs/analysis_logs/unquality_audios_<timestamp>.json`.

### 2. Audio Pairing

- By default, the script automatically picks the latest JSON file from `logs/analysis_logs`.
- You can specify a file explicitly using `--stats_json`.

```bash
python src/dataset/pair_audio.py
```

Results are saved to `logs/paired_audios/best_audio_pairs_<timestamp>.json`.

### 3. Create the Mixed Dataset

- By default, the script automatically picks the latest JSON file from `logs/paired_audios`.
- You can specify a file explicitly using `--pairs_json`.

```bash
python src/dataset/create_dataset.py
```

To also generate a validation split from the training data:

```bash
python src/dataset/create_dataset.py --create_valid
```

The dataset will be created at `data/datasets/<split>/{mix,s1,s2}`.

### 4. Fine-tune Conv-TasNet

```bash
python src/training/training.py --data_root data/datasets
```

If your dataset is not located at `data/datasets`, always pass `--data_root` with the correct path.

To disable Weights & Biases logging:

```bash
python src/training/training.py --data_root data/datasets --wandb_disabled
```

The best model checkpoint will be saved to `models/finetuned/convtasnet_best.pth`.

## Audio Quality Configuration

- `configs/quality_required.py` is where quality thresholds are defined.
- `SILENCE_THRESHOLD`: mean amplitude threshold below which audio is considered silent.
- `SHORT_DURATION_THRESHOLD`: minimum acceptable duration in seconds.

## Audio Configuration

- `configs/audio_config.py` keeps sample rate and crop/pad settings consistent between dataset generation and training.

## Project Structure

```
configs/
  audio_config.py
  quality_required.py
data/
  raw/              # original VIVOS data
  datasets/         # mixed dataset (output)
logs/
  analysis_logs/
  paired_audios/
models/
  finetuned/
src/
  analysis/quality_check.py
  dataset/
    pair_audio.py
    create_dataset.py
  training/
    dataset.py
    loss.py
    training.py
```
