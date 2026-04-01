import os
import json
import random
import argparse
import sys
from pathlib import Path
import torch
import torchaudio
import pyloudnorm as pyln

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs.audio_config import TARGET_SR, CROP_OR_PAD_MODE

# ==============================
# CONFIG
# ==============================
ROOT_TRAIN = r"data/raw/vivos/train/waves"
ROOT_TEST = r"data/raw/vivos/test/waves"

OUTPUT_DIR = "data/datasets"


# ==============================
# ARGUMENTS
# ==============================

parser = argparse.ArgumentParser()
parser.add_argument("--create_valid", action="store_true")
parser.add_argument(
    "--pairs_json",
    type=str,
    default=None,
    help="Path to pairs JSON file. If not set, uses latest JSON in logs/paired_audios.",
)
args = parser.parse_args()


# ==============================
# UTILITIES
# ==============================

def get_latest_json(log_dir):
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    candidates = [
        os.path.join(log_dir, f)
        for f in os.listdir(log_dir)
        if f.lower().endswith(".json")
    ]

    if not candidates:
        raise FileNotFoundError(f"No JSON files found in {log_dir}")

    return max(candidates, key=os.path.getmtime)


# ==============================
# AUDIO UTILS
# ==============================

def loudness_normalize(waveform, sr, target_lufs):

    meter = pyln.Meter(sr)

    audio = waveform.numpy()

    loudness = meter.integrated_loudness(audio)

    normalized = pyln.normalize.loudness(audio, loudness, target_lufs)

    return torch.tensor(normalized)


def crop_or_pad(w1, w2, mode):

    len1 = w1.shape[-1]
    len2 = w2.shape[-1]

    if mode == "min":

        target = min(len1, len2)

        w1 = w1[..., :target]
        w2 = w2[..., :target]

    else:

        target = max(len1, len2)

        if len1 < target:
            w1 = torch.nn.functional.pad(w1, (0, target - len1))

        if len2 < target:
            w2 = torch.nn.functional.pad(w2, (0, target - len2))

    return w1, w2


# ==============================
# DATASET UTILS
# ==============================

def create_split_dirs(split):

    for sub in ["mix", "s1", "s2"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, sub), exist_ok=True)


def load_pairs():

    pairs_json = args.pairs_json or get_latest_json("logs/paired_audios")
    print(f"Using pairs JSON: {pairs_json}")

    with open(pairs_json, "r") as f:
        data = json.load(f)

    train_pairs = data["datasets"]["train"]["pairs"]
    test_pairs = data["datasets"]["test"]["pairs"]

    return train_pairs, test_pairs


# ==============================
# SPLIT LOGIC
# ==============================

def prepare_splits(train_pairs, test_pairs):

    if not args.create_valid:

        return {
            "train": train_pairs,
            "test": test_pairs
        }

    random.shuffle(train_pairs)

    n_valid = int(len(train_pairs) * 0.05)

    new_test = train_pairs[:n_valid]
    new_train = train_pairs[n_valid:]

    valid = test_pairs

    return {
        "train": new_train,
        "test": new_test,
        "valid": valid
    }


# ==============================
# PROCESS ONE PAIR
# ==============================

def process_pair(pair, root_dir):

    file1 = pair["file1"]
    file2 = pair["file2"]

    spk1 = pair["speaker1"]
    spk2 = pair["speaker2"]

    path1 = os.path.join(root_dir, spk1, file1)
    path2 = os.path.join(root_dir, spk2, file2)

    w1, sr1 = torchaudio.load(path1)
    w2, sr2 = torchaudio.load(path2)

    w1 = w1.squeeze(0)
    w2 = w2.squeeze(0)

    if sr1 != TARGET_SR:
        w1 = torchaudio.transforms.Resample(sr1, TARGET_SR)(w1)

    if sr2 != TARGET_SR:
        w2 = torchaudio.transforms.Resample(sr2, TARGET_SR)(w2)

    w1, w2 = crop_or_pad(w1, w2, CROP_OR_PAD_MODE)

    lufs1 = random.uniform(-33, -25)
    lufs2 = random.uniform(-33, -25)

    w1 = loudness_normalize(w1, TARGET_SR, lufs1)
    w2 = loudness_normalize(w2, TARGET_SR, lufs2)

    mixture = w1 + w2

    peak = torch.max(torch.abs(mixture))

    if peak > 1:
        mixture = mixture / peak
        w1 = w1 / peak
        w2 = w2 / peak

    return mixture, w1, w2


# ==============================
# PROCESS DATASET
# ==============================

def process_dataset(pairs, root_dir, split):

    create_split_dirs(split)

    print(f"\nProcessing {split} set: {len(pairs)} samples")

    for idx, pair in enumerate(pairs):

        mixture, w1, w2 = process_pair(pair, root_dir)

        name = f"mix_{idx:05d}.wav"

        torchaudio.save(
            os.path.join(OUTPUT_DIR, split, "mix", name),
            mixture.unsqueeze(0),
            TARGET_SR,
            encoding="PCM_S",
            bits_per_sample=16
        )

        torchaudio.save(
            os.path.join(OUTPUT_DIR, split, "s1", name),
            w1.unsqueeze(0),
            TARGET_SR,
            encoding="PCM_S",
            bits_per_sample=16
        )

        torchaudio.save(
            os.path.join(OUTPUT_DIR, split, "s2", name),
            w2.unsqueeze(0),
            TARGET_SR,
            encoding="PCM_S",
            bits_per_sample=16
        )

        if idx % 100 == 0:
            print(f"{split}: processed {idx}")


# ==============================
# MAIN
# ==============================

def main():

    train_pairs, test_pairs = load_pairs()

    splits = prepare_splits(train_pairs, test_pairs)

    for split, pairs in splits.items():

        if split == "train":
            root = ROOT_TRAIN

        elif split == "test":
            root = ROOT_TRAIN if args.create_valid else ROOT_TEST

        else:
            root = ROOT_TEST

        process_dataset(pairs, root, split)

    print("\nAll datasets generated successfully!")


if __name__ == "__main__":
    main()
