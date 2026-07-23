"""
dataloader.py
=============

Handles batching, variable-length sequence padding for landmarks (pose, face, hands) 
and text targets (input_ids, attention_mask).
"""

from __future__ import annotations

from typing import Any, List, Dict

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from torch.nn.utils.rnn import pad_sequence
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader


class SignLanguageCollate:
    """
    Collate function to handle variable-length padding for landmark sequences
    and target token IDs.
    """
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        landmarks = [item["landmarks"] for item in batch]
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        sentences = [item["sentence"] for item in batch]
        glosses = [item["gloss"] for item in batch]
        video_names = [item["video_name"] for item in batch]

        # Landmark sequences length tracking
        landmark_lengths = torch.tensor([l.shape[0] for l in landmarks], dtype=torch.long)
        
        # Pad landmarks to max sequence length in batch: (Batch, max_T, feature_dim)
        padded_landmarks = pad_sequence(landmarks, batch_first=True, padding_value=0.0)

        # Generate landmark mask: 1 for active frames, 0 for padded frames
        max_T = padded_landmarks.shape[1]
        landmark_mask = torch.zeros(len(landmarks), max_T, dtype=torch.float32)
        for idx, length in enumerate(landmark_lengths):
            landmark_mask[idx, :length] = 1.0

        # Pad text target token ids with the pad token ID: (Batch, max_tgt_len)
        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        
        # Pad text attention masks with 0 (since they represent padding): (Batch, max_tgt_len)
        padded_attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        return {
            "landmarks": padded_landmarks,
            "landmark_lengths": landmark_lengths,
            "landmark_mask": landmark_mask,
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "sentences": sentences,
            "glosses": glosses,
            "video_names": video_names
        }


def get_dataloader(
    dataset: Any,
    batch_size: int,
    pad_token_id: int,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    Creates a DataLoader wrapper for SignLanguageDataset.
    """
    collate_fn = SignLanguageCollate(pad_token_id)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
