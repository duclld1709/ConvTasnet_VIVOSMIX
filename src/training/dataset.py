import os
import random
import sys
from pathlib import Path
from typing import List, Tuple, Union

import torch
from torch.utils.data import Dataset
from torchaudio.datasets.utils import _load_waveform
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs.audio_config import TARGET_SR, TRAINING_CROP_SECONDS


class VIVOSMIX(Dataset):
    """
    VIVOSMIX Speech Separation Dataset

    Folder structure:

    root/
        train/
            mix/
            s1/
            s2/
        test/
            mix/
            s1/
            s2/
    """

    def __init__(
        self,
        root: Union[str, Path],
        subset: str = "train",
        num_speakers: int = 2,
        sample_rate: int = TARGET_SR,
    ):
        self.root = Path(root)
        self.subset = subset
        self.sample_rate = sample_rate
        self.num_speakers = num_speakers

        self.mix_dir = self.root / subset / "mix"

        if not self.mix_dir.exists():
            raise RuntimeError(f"{self.mix_dir} does not exist")

        self.src_dirs = [
            self.root / subset / f"s{i+1}"
            for i in range(num_speakers)
        ]

        self.files = sorted([p.name for p in self.mix_dir.glob("*.wav")])

    def __len__(self):
        return len(self.files)

    def get_metadata(self, key: int):

        filename = self.files[key]

        mixed_path = os.path.relpath(self.mix_dir / filename, self.root)

        src_paths = []
        for src_dir in self.src_dirs:
            src_paths.append(os.path.relpath(src_dir / filename, self.root))

        return self.sample_rate, mixed_path, src_paths

    def _load_sample(self, key):

        metadata = self.get_metadata(key)

        mixed = _load_waveform(self.root, metadata[1], metadata[0])

        srcs = []
        for i, path in enumerate(metadata[2]):

            src = _load_waveform(self.root, path, metadata[0])

            if mixed.shape != src.shape:
                raise ValueError(
                    f"Different waveform shapes. mixed: {mixed.shape}, src[{i}]: {src.shape}"
                )

            srcs.append(src)

        return self.sample_rate, mixed, srcs

    def __getitem__(self, key):

        return self._load_sample(key)
    

def pad_collate_fn(batch):
    # batch: list of (sample_rate, mixture[1,T], [src1[1,T], src2[1,T]])
    sample_rates, mixtures, sources_list = zip(*batch)
    
    TARGET_LEN = int(TRAINING_CROP_SECONDS * sample_rates[0])
    
    padded_mixtures = []
    padded_sources  = []
    
    for m, srcs in zip(mixtures, sources_list):
        T = m.shape[-1]
        
        if T < TARGET_LEN:
            # Pad to TARGET_LEN
            pad_amount = TARGET_LEN - T
            padded_mixtures.append(F.pad(m, (0, pad_amount)))
            s = torch.stack(srcs, dim=0).squeeze(1)
            padded_sources.append(F.pad(s, (0, pad_amount)))
        else:
            # Random crop to TARGET_LEN
            max_start = T - TARGET_LEN
            start = random.randint(0, max_start)
            end = start + TARGET_LEN
            
            padded_mixtures.append(m[..., start:end])
            s = torch.stack(srcs, dim=0).squeeze(1)
            padded_sources.append(s[..., start:end])
    
    mixtures_batch = torch.stack(padded_mixtures)  # [B, 1, TARGET_LEN]
    sources_batch  = torch.stack(padded_sources)   # [B, N_src, TARGET_LEN]
    return mixtures_batch, sources_batch

# def pad_collate_fn(batch):
#     """
#     batch là list các output từ VIVOSMIX.__getitem__:
#     [(sr, mixture, [s1, s2]), (sr, mixture, [s1, s2]), ...]
#     """
#     # 1. Unpack dữ liệu từ batch
#     # mixtures: list các tensor [1, T]
#     # sources_list: list các list [tensor[1, T], tensor[1, T]]
#     sample_rates, mixtures, sources_list = zip(*batch)

#     # 2. Tìm độ dài lớn nhất (Time steps) trong batch này để padding
#     max_len = max([m.shape[-1] for m in mixtures])

#     padded_mixtures = []
#     padded_sources = []

#     for i in range(len(mixtures)):
#         m = mixtures[i]        # Shape: [1, T]
#         srcs = sources_list[i] # List gồm [s1, s2], mỗi cái shape [1, T]
        
#         # Tính toán lượng padding cần thiết
#         pad_amount = max_len - m.shape[-1]
        
#         # --- Xử lý Mixture ---
#         # Pad mixture từ [1, T] -> [1, max_len]
#         m_padded = F.pad(m, (0, pad_amount))
#         padded_mixtures.append(m_padded)
        
#         # --- Xử lý Sources ---
#         # Bước A: Gộp list các source [1, T] thành 1 tensor duy nhất [Num_Sources, T]
#         # VIVOSMIX trả về list các tensor [1, T], ta dùng squeeze(1) để bỏ dimension channel 
#         # sau đó stack lại để được [2, T]
#         s_combined = torch.cat(srcs, dim=0) # Kết quả: [2, T]
        
#         # Bước B: Pad sources từ [2, T] -> [2, max_len]
#         s_padded = F.pad(s_combined, (0, pad_amount))
#         padded_sources.append(s_padded)

#     # 3. Stack lại thành Batch tensor
#     # mixtures_batch: [Batch, 1, max_len] -> Khớp đầu vào ConvTasNet
#     # sources_batch:  [Batch, 2, max_len] -> Khớp để tính PIT Loss
#     mixtures_batch = torch.stack(padded_mixtures)
#     sources_batch = torch.stack(padded_sources)

#     return mixtures_batch, sources_batch
