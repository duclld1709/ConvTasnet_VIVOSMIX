import os
import json
import random
import argparse
import torchaudio

from datetime import datetime

# ------------------------------------------------
# Create timestamp
# ------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_JSON = f"logs/paired_audios/best_audio_pairs_{timestamp}.json"
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

DATASETS = {
        "train": "data/raw/vivos/train/waves",
        "test": "data/raw/vivos/test/waves"
}

ITERATIONS = 100


# ------------------------------------------------
# ARGUMENTS
# ------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Pair VIVOS audio files by similar duration")
    parser.add_argument(
        "--stats_json",
        type=str,
        default=None,
        help="Path to JSON stats file. If not set, uses latest JSON in logs/analysis_logs.",
    )
    return parser.parse_args()


# ------------------------------------------------
# UTILITIES
# ------------------------------------------------
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


# ------------------------------------------------
# Load invalid files
# ------------------------------------------------
def load_invalid_files(stats, split):

    short_files = set(x["file"] for x in stats[split]["short_audio_files"])

    silent_files = set()
    if "silent_audio_files" in stats[split]:
        silent_files = set(x["file"] for x in stats[split]["silent_audio_files"])

    return short_files.union(silent_files)


# ------------------------------------------------
# Collect valid audio
# ------------------------------------------------
def collect_audio(waves_root, invalid_files):

    audio_list = []

    for speaker in os.listdir(waves_root):

        speaker_dir = os.path.join(waves_root, speaker)

        if not os.path.isdir(speaker_dir):
            continue

        for file in os.listdir(speaker_dir):

            if not file.endswith(".wav"):
                continue

            if file in invalid_files:
                continue

            path = os.path.join(speaker_dir, file)

            waveform, sr = torchaudio.load(path)
            duration = waveform.shape[1] / sr

            audio_list.append({
                "file": file,
                "speaker": speaker,
                "duration": duration
            })

    return audio_list


# ------------------------------------------------
# Pairing
# ------------------------------------------------
def create_pairs(audio_data):

    data = audio_data.copy()
    random.shuffle(data)

    pairs = []
    i = 0

    while i < len(data) - 1:

        a = data[i]
        b = data[i + 1]

        if a["speaker"] == b["speaker"]:

            found = False

            for j in range(i + 2, len(data)):
                if data[j]["speaker"] != a["speaker"]:
                    data[i + 1], data[j] = data[j], data[i + 1]
                    b = data[i + 1]
                    found = True
                    break

            if not found:
                i += 1
                continue

        diff = abs(a["duration"] - b["duration"])

        pairs.append({
            "file1": a["file"],
            "speaker1": a["speaker"],
            "duration1": a["duration"],
            "file2": b["file"],
            "speaker2": b["speaker"],
            "duration2": b["duration"],
            "duration_diff": diff
        })

        i += 2

    # odd case
    if len(data) % 2 == 1:

        last = data[-1]

        for p in pairs:
            if p["speaker1"] != last["speaker"]:

                diff = abs(last["duration"] - p["duration1"])

                pairs.append({
                    "file1": last["file"],
                    "speaker1": last["speaker"],
                    "duration1": last["duration"],
                    "file2": p["file1"],
                    "speaker2": p["speaker1"],
                    "duration2": p["duration1"],
                    "duration_diff": diff
                })

                break

    total_diff = sum(p["duration_diff"] for p in pairs)

    return pairs, total_diff


# ------------------------------------------------
# Optimize pairing
# ------------------------------------------------
def find_best_pairs(audio_list):

    best_pairs = None
    best_diff = float("inf")

    for i in range(ITERATIONS):

        pairs, total_diff = create_pairs(audio_list)

        if total_diff < best_diff:

            best_diff = total_diff
            best_pairs = pairs

            print(f"Iteration {i} -> New BEST diff: {best_diff:.2f}")

    return best_pairs, best_diff


# ------------------------------------------------
def main():
    args = parse_args()

    stats_json = args.stats_json or get_latest_json("logs/analysis_logs")
    print(f"Using stats JSON: {stats_json}")

    with open(stats_json, "r", encoding="utf-8") as f:
        stats = json.load(f)

    final_result = {}

    for split, path in DATASETS.items():

        print("\n==============================")
        print("Processing:", split.upper())
        print("==============================")

        invalid_files = load_invalid_files(stats, split)

        audio_list = collect_audio(path, invalid_files)

        print("Valid audio:", len(audio_list))

        best_pairs, best_diff = find_best_pairs(audio_list)

        final_result[split] = {
            "total_pairs": len(best_pairs),
            "total_duration_difference": best_diff,
            "pairs": best_pairs
        }

        print("Best duration difference:", best_diff)

    # ------------------------------------------------
    # Save
    # ------------------------------------------------
    result = {
        "iterations": ITERATIONS,
        "datasets": final_result
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print("\nSaved to:", OUTPUT_JSON)


if __name__ == "__main__":
    main()
