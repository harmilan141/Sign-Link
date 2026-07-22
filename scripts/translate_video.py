"""
translate_video.py
==================

Translates a raw input sign language video (.mp4) into an English sentence.
Steps:
1. Extract landmarks frame-by-frame using MediaPipe Holistic.
2. Initialize and load the trained SignTranslationModel checkpoint.
3. Pass the landmark sequence through the model and decode the translation using beam search.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import numpy as np

import torch
from transformers import MBart50TokenizerFast

# Add preprocessing folder to sys.path to import mediapipe_utils
sys.path.append(str(Path(__file__).parent.resolve() / "preprocessing"))
try:
    from mediapipe_utils import initialize_holistic, extract_video_landmark_sequence
except ImportError:
    # Fallback if running from a different working directory
    sys.path.append(os.path.join(os.path.dirname(__file__), "preprocessing"))
    from mediapipe_utils import initialize_holistic, extract_video_landmark_sequence

from model import SignTranslationModel

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("translate")


def main():
    parser = argparse.ArgumentParser(description="Translate a raw sign language video to English.")
    parser.add_argument("--video", type=Path, required=True, help="Path to raw sign language video (.mp4)")
    parser.add_argument("--checkpoint", type=Path, default=Path("/workspace/SignLink/models/checkpoint_best.pt"), help="Path to best model checkpoint")
    parser.add_argument("--tokenizer-name", type=str, default="facebook/mbart-large-50", help="Pretrained tokenizer name")
    parser.add_argument("--model-name", type=str, default="facebook/mbart-large-50", help="Pretrained mbart model name")
    parser.add_argument("--num-beams", type=int, default=4, help="Beam size for decoding")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum length of translated output sentence")
    
    args = parser.parse_args()

    if not args.video.is_file():
        logger.error("Input video file not found: %s", args.video)
        sys.exit(1)

    if not args.checkpoint.is_file():
        logger.error("Checkpoint file not found: %s", args.checkpoint)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # 1. Load Tokenizer
    logger.info("Loading tokenizer %s", args.tokenizer_name)
    tokenizer = MBart50TokenizerFast.from_pretrained(args.tokenizer_name, src_lang="hi_IN", tgt_lang="en_XX")

    # 2. Extract Landmarks using MediaPipe Holistic
    logger.info("Extracting landmarks from video using MediaPipe Holistic: %s", args.video.name)
    
    # Initialize holistic model
    holistic = initialize_holistic(static_image_mode=False, model_complexity=1)
    
    try:
        landmarks = extract_video_landmark_sequence(args.video, holistic)
        fps = 0.0
        total_frames = landmarks.shape[0]
        logger.info(f"Landmarks successfully extracted. Frames: {total_frames} | Shape: {landmarks.shape}")
    except Exception as e:
        logger.error("Error extracting landmarks from video: %s", e)
        sys.exit(1)
    finally:
        holistic.close()

    # Normalize landmarks type to float32 and add batch dimension
    landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, 1662)
    landmark_mask = torch.ones(1, landmarks.shape[0], dtype=torch.float32).to(device)       # (1, T)

    # 3. Load Trained Model Checkpoint
    logger.info("Loading model checkpoint from %s", args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Auto-detect input dimension
    input_dim = landmarks.shape[1]
    
    model = SignTranslationModel(input_dim=input_dim, mbart_model_name=args.model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # 4. Generate Translation using Beam Search
    logger.info("Translating...")
    with torch.no_grad():
        generated_ids = model.generate(
            landmarks=landmarks_tensor,
            landmark_mask=landmark_mask,
            tokenizer=tokenizer,
            num_beams=args.num_beams,
            max_length=args.max_length
        )
        
    translation = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    print("\n" + "=" * 60)
    print("TRANSLATION RESULT:")
    print("-" * 60)
    print(translation)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
