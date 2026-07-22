"""
dataset.py
==========

Custom PyTorch Dataset class for Sign Language Translation. Loads preprocessed 
compressed landmark sequences (.npz files) and tokenizes target sentences.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Ensure dataset can load local preprocessing and data submodules
scripts_dir = Path(__file__).parent.resolve()
if str(scripts_dir) not in sys.path:
    sys.path.append(str(scripts_dir))

import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.normalize_landmarks import normalize_landmarks_sequence, get_visibility_mask
from data.augmentations import augment_landmarks_sequence, get_augmented_target_sentence

logger = logging.getLogger(__name__)


class SignLanguageDataset(Dataset):
    """
    Loads preprocessed landmark sequence files and target sentences.
    """
    def __init__(
        self,
        npz_dir: str | Path,
        tokenizer: Any,
        max_seq_len: int = 512,
        max_tgt_len: int = 128,
        augment_config: dict | None = None,
        paraphrase_hook: callable | None = None,
        use_normalization: bool = True,
        use_masking: bool = True
    ) -> None:
        """
        Args:
            npz_dir: Directory containing preprocessed .npz files.
            tokenizer: Pre-initialized HuggingFace tokenizer (e.g. MBart50Tokenizer).
            max_seq_len: Maximum sequence length for padding/truncating landmarks.
            max_tgt_len: Maximum length for target English sentence tokenization.
            augment_config: Dictionary containing augmentation probabilities and limits.
            paraphrase_hook: Function for target sentence paraphrasing.
            use_normalization: Whether to normalize coordinate values torso-relatively.
            use_masking: Whether to extract detection occlusion masks.
        """
        self.npz_dir = Path(npz_dir)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_tgt_len = max_tgt_len
        self.augment_config = augment_config
        self.paraphrase_hook = paraphrase_hook
        self.use_normalization = use_normalization
        self.use_masking = use_masking
        
        # Read files and skip any empty/broken archives
        self.files = sorted([
            p for p in self.npz_dir.glob("*.npz")
            if p.is_file() and p.stat().st_size > 0
        ])
        
        if not self.files:
            logger.warning("No preprocessed .npz archives found in %s", npz_dir)
        else:
            logger.info("Loaded %d landmark sequences from %s", len(self.files), self.npz_dir)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a single dataset item.
        """
        file_path = self.files[idx]
        
        try:
            # allow_pickle=False for security
            with np.load(file_path, allow_pickle=False) as data:
                landmarks = np.asarray(data["landmarks"], dtype=np.float32)
                sentence = str(data["sentence"])
                gloss = str(data["gloss"])
        except Exception:
            logger.exception("Error loading NPZ file: %s", file_path)
            # Return zeroed dummy data if corrupted
            landmarks = np.zeros((1, 1662), dtype=np.float32)
            sentence = ""
            gloss = ""

        # Enforce maximum frame sequence length
        if landmarks.shape[0] > self.max_seq_len:
            landmarks = landmarks[:self.max_seq_len]

        # Apply data augmentations (only applied if augment_config is provided/training)
        if self.augment_config is not None:
            landmarks = augment_landmarks_sequence(landmarks, self.augment_config)
            sentence = get_augmented_target_sentence(sentence, self.paraphrase_hook)

        # Apply spatial landmark normalization if enabled
        if self.use_normalization:
            landmarks = normalize_landmarks_sequence(landmarks)

        # Extract explicit detection masks (Pose, Left Hand, Right Hand) if enabled
        if self.use_masking:
            mask = get_visibility_mask(landmarks)
            # Append detection masks as explicit features (making landmark vector 1665 dimensions)
            landmarks = np.concatenate([landmarks, mask], axis=-1)

        landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32)

        # Tokenize sentence (without padding; padding is done inside collate_fn)
        tokenized_tgt = self.tokenizer(
            sentence,
            max_length=self.max_tgt_len,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )

        input_ids = tokenized_tgt["input_ids"].squeeze(0)
        attention_mask = tokenized_tgt["attention_mask"].squeeze(0)

        return {
            "landmarks": landmarks_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sentence": sentence,
            "gloss": gloss,
            "video_name": file_path.stem
        }
