import json
import math
import os
from typing import Dict, List

import librosa
import numpy as np


# --- Constants ---
DATASET_PATH = "data/genres_original"
JSON_PATH = "data.json"
SAMPLE_RATE = 22050
DURATION = 30  # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
DEFAULT_NUM_SEGMENTS = 10


def prepare_dataset(
    dataset_path: str,
    json_path: str,
    n_mels: int = 128,
    hop_length: int = 512,
    n_fft: int = 2048,
    num_segments: int = DEFAULT_NUM_SEGMENTS,
) -> None:
    """
    Extract Mel-spectrogram features (and associated metadata) from the GTZAN dataset
    and save them to JSON.

    Args:
        dataset_path: Path to the GTZAN dataset folder containing genre subfolders.
        json_path: Path to the output JSON file.
        n_mels: Number of Mel bands to generate.
        hop_length: Number of samples between successive frames.
        n_fft: Length of the FFT window.
        num_segments: Number of non-overlapping segments per audio track.
            Metadata per segment includes the source filename and segment index.
    """

    data: Dict[str, List] = {
        "mapping": [],
        "mfcc": [],
        "labels": [],
        "files": [],
        "segment_indices": [],
    }
    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_vectors = math.ceil(num_samples_per_segment / hop_length)

    print("Processing dataset...")
    for index, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath == dataset_path:
            continue

        genre_label = os.path.basename(dirpath)
        data["mapping"].append(genre_label)
        print(f"\nProcessing {genre_label}")

        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                signal, sr = librosa.load(
                    file_path, sr=SAMPLE_RATE, duration=DURATION, mono=True
                )

                if signal.size < SAMPLES_PER_TRACK:
                    signal = librosa.util.fix_length(signal, size=SAMPLES_PER_TRACK)

                for segment in range(num_segments):
                    start_sample = num_samples_per_segment * segment
                    end_sample = start_sample + num_samples_per_segment

                    if end_sample > signal.size:
                        continue

                    segment_signal = signal[start_sample:end_sample]
                    mel_spectrogram = librosa.feature.melspectrogram(
                        y=segment_signal,
                        sr=sr,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        n_mels=n_mels,
                    )
                    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

                    if mel_spectrogram_db.shape[1] == expected_num_vectors:
                        data["mfcc"].append(mel_spectrogram_db.tolist())
                        data["labels"].append(len(data["mapping"]) - 1)
                        data["files"].append(os.path.relpath(file_path, dataset_path))
                        data["segment_indices"].append(segment)
                        print(f"{file_path}, segment {segment + 1}", end="\r")

            except Exception as exc:
                print(f"Could not process {file_path}: {exc}")

    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4)

    print("\n...Done!")


if __name__ == "__main__":
    prepare_dataset(DATASET_PATH, JSON_PATH)

