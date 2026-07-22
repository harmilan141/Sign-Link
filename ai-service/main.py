import asyncio
import base64
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import threading
import time
import uuid
import shutil
from typing import Any, Dict, List
import tempfile

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
import sys
from pathlib import Path

# Add project root and scripts paths
sys.path.append("/workspace/SignLink")
try:
    from model import SignTranslationModel
    from preprocessing.mediapipe_utils import initialize_holistic as init_holistic_mp, extract_frame_features, extract_video_landmark_sequence
    from preprocessing.normalize_landmarks import get_visibility_mask
except ImportError:
    # Fallback for local Windows environment
    sys.path.append(str(Path(__file__).parent.parent / "scripts"))
    sys.path.append(str(Path(__file__).parent.parent))
    from model import SignTranslationModel
    from preprocessing.mediapipe_utils import initialize_holistic as init_holistic_mp, extract_frame_features, extract_video_landmark_sequence
    from preprocessing.normalize_landmarks import get_visibility_mask

class FrameRequest(BaseModel):
    image: str

class SessionFrameRequest(BaseModel):
    session_id: str
    image: str

class SessionControlRequest(BaseModel):
    session_id: str

app = FastAPI(title="Sign Link AI & Landmark Service", version="1.0.0")

# Setup CORS
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "LANDMARK_CORS_ORIGINS",
        ",".join([
            "http://localhost:5173", "http://localhost:5174", "http://localhost:5175",
            "https://localhost:5173", "https://localhost:5174", "https://localhost:5175",
            "http://127.0.0.1:5173", "http://127.0.0.1:5174",
            "https://172.16.224.122:5173"
        ]),
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all during development/testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize standard models for legacy endpoint
hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.45,
    min_tracking_confidence=0.55,
)
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.5,
)

# AI Model Loader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
tokenizer = None

checkpoint_paths = [
    Path("/workspace/SignLink/models/A7_production/checkpoint_best.pt"),
    Path("/workspace/SignLink/models/A5_production/checkpoint_best.pt"),
    Path("/workspace/SignLink/models/checkpoint_best.pt"),
    Path(__file__).parent.parent / "models" / "checkpoint_best.pt",
    Path("./models/checkpoint_best.pt"),
    Path("../models/checkpoint_best.pt")
]

checkpoint_path = None
for p in checkpoint_paths:
    if p.is_file():
        checkpoint_path = p
        break

if checkpoint_path:
    try:
        from transformers import MBart50TokenizerFast
        print(f"[AI Model] Loading checkpoint from {checkpoint_path}")
        tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="hi_IN", tgt_lang="en_XX", local_files_only=True)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]
        
        # Check if modality encoders are used
        use_modality_encoders = "pose_enc.weight" in state_dict
        
        # Check decoder adaptation mode
        if any("lora_A" in k for k in state_dict.keys()):
            decoder_adapt_mode = "lora_cross_attn"
        else:
            decoder_adapt_mode = "frozen"
            
        # Check if ctc head is used and get vocab size
        use_auxiliary_ctc = "ctc_head.weight" in state_dict
        gloss_vocab_size = None
        if use_auxiliary_ctc:
            gloss_vocab_size = state_dict["ctc_head.weight"].shape[0]
            
        # Check input dim if flat projection is used
        input_dim = 1665
        if "linear_in.weight" in state_dict:
            input_dim = state_dict["linear_in.weight"].shape[1]
            
        model = SignTranslationModel(
            input_dim=input_dim,
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            mbart_model_name="facebook/mbart-large-50",
            use_modality_encoders=use_modality_encoders,
            decoder_adapt_mode=decoder_adapt_mode,
            lora_r=8,
            lora_alpha=16.0,
            use_auxiliary_ctc=use_auxiliary_ctc,
            gloss_vocab_size=gloss_vocab_size
        )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("[AI Model] Model loaded successfully on", device)
    except Exception as e:
        print(f"[AI Model] Error loading model: {e}")
else:
    print("[AI Model] No checkpoint found. Inference endpoints will return mock sentences.")

# ── LLM Grammatical Refinement (signlinkLLM) ────────────────────────────────
# Primary: Groq-hosted Llama-3.3-70B via langchain-groq (cloud, no GPU cost)
# Fallback: local Qwen 0.5B if GROQ_API_KEY is absent or Groq call fails
#
# System prompt and SignOutput schema ported from:
# https://github.com/Tarandeep98/signlinkLLM
# ─────────────────────────────────────────────────────────────────────────────

# ── Sign-language refinement prompt (from signlinkLLM/prompts.py) ────────────
SIGN_LLM_SYSTEM_PROMPT = """
You are an expert Sign Language Translation Assistant.

The input originates from sign-language recognition and may follow sign-language
grammar rather than English grammar.

Convert the input into natural, grammatically correct English while preserving
the original meaning.

Rules:
- Preserve meaning. Do not invent facts.
- Infer grammar, tense, articles, and pronouns only when necessary.
- Handle negation correctly.
- Handle commands and questions correctly.
- If multiple interpretations are possible, choose the most likely one and lower confidence.
- If the input is a single word, profanity, or unclear fragment, minimally correct it and set confidence to low.

Confidence:
- high: clear and unambiguous
- medium: required reasonable inference
- low: ambiguous, sparse, or unclear

Examples:

Input: me hungry
Output: I am hungry.

Input: i go market tomorrow
Output: I will go to the market tomorrow.

Input: mother hospital yesterday
Output: My mother went to the hospital yesterday.

Input: bring water
Output: Bring some water.

Input: water finish bring bottle
Output: The water is finished. Bring a bottle.
"""

# ── Groq / Llama-3.3-70B chain (signlinkLLM architecture) ───────────────────
groq_chain = None
try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import BaseModel as PydanticBase, Field

    class SignOutput(PydanticBase):
        """Structured output schema from signlinkLLM/models.py."""
        corrected_sentence: str = Field(description="Grammatically correct English sentence")
        confidence: str = Field(description="high | medium | low")

    groq_api_key = os.getenv("GROQ_API_KEY", "")
    if groq_api_key:
        _groq_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=groq_api_key
        ).with_structured_output(SignOutput)

        _groq_prompt = ChatPromptTemplate.from_messages([
            ("system", SIGN_LLM_SYSTEM_PROMPT),
            ("human", "{sign_text}")
        ])

        groq_chain = _groq_prompt | _groq_llm
        print("[SignLinkLLM] Groq / Llama-3.3-70B refinement chain ready.")
    else:
        print("[SignLinkLLM] GROQ_API_KEY not set — Groq refinement disabled. Set it to enable Llama-3.3-70B.")
except ImportError:
    print("[SignLinkLLM] langchain-groq not installed — run: pip install langchain-groq")
except Exception as _groq_err:
    print(f"[SignLinkLLM] Groq chain init failed: {_groq_err}")

# ── Local Qwen 0.5B fallback (existing behaviour) ────────────────────────────
llm_model = None
llm_tokenizer = None
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    llm_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"[LLM Fallback] Loading {llm_name} onto GPU...")
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_name, local_files_only=True)
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        local_files_only=True
    )
    print(f"[LLM Fallback] Qwen loaded on CUDA.")
except Exception as exc:
    print(f"[LLM Fallback] Could not load Qwen: {exc}")

# Active Sessions Storage
# Maps session_id -> { "features": list[np.ndarray], "holistic": HolisticInstance }
active_sessions: Dict[str, Dict[str, Any]] = {}
sessions_lock = threading.Lock()

# Helper Functions
def decode_image(data_url: str) -> np.ndarray:
    try:
        payload = data_url.split(",", 1)[1] if "," in data_url else data_url
        image_bytes = base64.b64decode(payload)
        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image") from exc

    if frame is None:
        raise HTTPException(status_code=400, detail="Unable to decode image")
    return frame

def resize_for_detection(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    max_w, max_h = 640, 480
    scale = min(max_w / width, max_h / height, 1.0)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

def landmark_to_dict(landmark: Any, index: int) -> dict[str, float | int]:
    return {
        "index": index,
        "x": float(landmark.x),
        "y": float(landmark.y),
        "z": float(landmark.z),
    }

# Legacy endpoints
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "sign-link-landmarks-and-translation"}

@app.post("/api/landmarks")
async def detect_landmarks(payload: FrameRequest) -> dict[str, Any]:
    # Legacy frame landmark detection used by WebRTC VideoCall
    frame = resize_for_detection(decode_image(payload.image))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = frame.shape[:2]
    
    # Run hand tracking
    hand_results = hands.process(rgb)
    detected_hands = []
    if hand_results.multi_hand_landmarks:
        handedness_list = hand_results.multi_handedness or []
        for hand_index, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            handedness = handedness_list[hand_index].classification[0].label if hand_index < len(handedness_list) else None
            detected_hands.append({
                "handedness": handedness,
                "landmarks": [landmark_to_dict(lm, idx) for idx, lm in enumerate(hand_landmarks.landmark)]
            })

    # Run face mesh
    face_results = face_mesh.process(rgb)
    detected_faces = []
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            detected_faces.append({
                "landmarks": [landmark_to_dict(lm, idx) for idx, lm in enumerate(face_landmarks.landmark)]
            })

    return {
        "image": {"width": width, "height": height},
        "hands": detected_hands,
        "faces": detected_faces,
        "status": {
            "handsDetected": len(detected_hands),
            "facesDetected": len(detected_faces),
        },
    }

# Translation Session Management
@app.post("/api/translate/session/start")
def start_session() -> dict[str, str]:
    session_id = str(uuid.uuid4())
    holistic_model = init_holistic_mp(static_image_mode=False)
    with sessions_lock:
        active_sessions[session_id] = {
            "features": [],
            "holistic": holistic_model
        }
    print(f"[Session] Started session: {session_id}")
    return {"session_id": session_id}

@app.post("/api/translate/session/frame")
async def process_session_frame(payload: SessionFrameRequest) -> dict[str, Any]:
    session_id = payload.session_id
    with sessions_lock:
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        session = active_sessions[session_id]
        
    frame = resize_for_detection(decode_image(payload.image))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = frame.shape[:2]
    
    # Process using session's holistic model
    # To prevent parallel process execution blocks, we run in the thread pool
    results = await run_in_threadpool(session["holistic"].process, rgb)
    
    # Extract, normalize, and flatten features for the model sequence
    from preprocessing.mediapipe_utils import extract_landmarks as extract_lms, normalize_landmarks as norm_lms, flatten_landmarks as flat_lms
    landmarks_dict = extract_lms(results)
    normalized_dict = norm_lms(landmarks_dict)
    feature_vector = flat_lms(normalized_dict)
    
    session["features"].append(feature_vector)
    
    # Format hands/faces for drawing overlay
    detected_hands = []
    if results.left_hand_landmarks:
        detected_hands.append({
            "handedness": "Left",
            "landmarks": [landmark_to_dict(lm, idx) for idx, lm in enumerate(results.left_hand_landmarks.landmark)]
        })
    if results.right_hand_landmarks:
        detected_hands.append({
            "handedness": "Right",
            "landmarks": [landmark_to_dict(lm, idx) for idx, lm in enumerate(results.right_hand_landmarks.landmark)]
        })

    detected_faces = []
    if results.face_landmarks:
        detected_faces.append({
            "landmarks": [landmark_to_dict(lm, idx) for idx, lm in enumerate(results.face_landmarks.landmark)]
        })

    return {
        "image": {"width": width, "height": height},
        "hands": detected_hands,
        "faces": detected_faces,
        "status": {
            "handsDetected": len(detected_hands),
            "facesDetected": len(detected_faces),
        },
    }

@app.post("/api/translate/session/stop")
async def stop_session(payload: SessionControlRequest) -> dict[str, str]:
    session_id = payload.session_id
    with sessions_lock:
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        session = active_sessions.pop(session_id)
        
    # Close session holistic instance
    session["holistic"].close()
    
    features = session["features"]
    print(f"[Session] Stopping session {session_id}. Frame count: {len(features)}")
    
    if len(features) < 6:
        return {"translation": "Gesture too short. Please sign for at least 1-2 seconds."}
        
    # Check if hands were detected in any of the frames
    landmarks = np.stack(features, axis=0) # (T, 1662)
    hands_data = landmarks[:, 1536:1662] # Left hand: 1536-1599, Right hand: 1599-1662
    if np.all(hands_data == 0.0):
        print("[AI Session] No hands detected in captured sequence. Skipping inference.")
        return {"translation": "No hands detected. Please make sure your hands are visible."}

    translation = run_inference_on_features(features)
    return {"translation": translation}

# Video File translation
@app.post("/api/translate/video")
async def translate_video_file(file: UploadFile = File(...)) -> dict[str, str]:
    # Save UploadFile to temp file
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / file.filename
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"[Video Translate] Uploaded video file: {file.filename}")
        
        # Run landmark extraction on video file
        # To avoid blocking, run in thread pool
        features = await run_in_threadpool(
            extract_video_landmark_sequence, 
            str(temp_path), 
            None, 
            True, 
            512
        )
        
        if features.shape[0] == 0:
            return {"translation": "Unable to extract landmark sequences from video."}
            
        # Check if hands were detected in the video
        hands_data = features[:, 1536:1662]
        if np.all(hands_data == 0.0):
            print("[Video Translate] No hands detected in video. Skipping inference.")
            return {"translation": "No hands detected. Please make sure hands are visible in the video."}
            
        # Run PyTorch Model Inference
        translation = run_inference_on_features(list(features))
        return {"translation": translation}
        
    finally:
        # Clean up temp file
        shutil.rmtree(temp_dir)

def run_inference_on_features(features: List[np.ndarray]) -> str:
    # Handle mock translation if model is not loaded
    if model is None or tokenizer is None:
        mock_translations = [
            "what is your name",
            "where is the train station",
            "can you repeat that please",
            "how can i help you",
            "are you free today",
            "please sit down"
        ]
        return np.random.choice(mock_translations)
        
    # If the user signs for a very long time, downsample the sequence evenly
    # instead of hard-truncating, so the model sees the full context from start to end.
    max_original_frames = 250
    if len(features) > max_original_frames:
        print(f"[AI Inference] Sequence too long ({len(features)} frames). Downsampling to {max_original_frames} to preserve full context.")
        indices = np.linspace(0, len(features) - 1, max_original_frames)
        downsampled = []
        for idx in indices:
            low = int(np.floor(idx))
            high = int(np.ceil(idx))
            weight = idx - low
            downsampled.append(features[low] * (1.0 - weight) + features[high] * weight)
        features = downsampled
        
    T = len(features)
    # Resample captured sequence from ~6.67 fps (150ms interval) to 25 fps
    # to match the video speed the model was trained on (3.75x speed stretch)
    target_len = int(round(T * 3.75))
    if target_len < 12:
        target_len = 12
    
    indices = np.linspace(0, T - 1, target_len)
    resampled_features = []
    for idx in indices:
        low = int(np.floor(idx))
        high = int(np.ceil(idx))
        weight = idx - low
        resampled_features.append(features[low] * (1.0 - weight) + features[high] * weight)
        
    landmarks = np.stack(resampled_features, axis=0) # (target_len, 1662)
    
    # ── CRITICAL: append 3 detection-mask columns (dims 1662–1664) ────────────
    # The model's pose_enc, left_hand_enc, right_hand_enc each concatenate a
    # 1-dim presence mask to the raw coordinates (see model.py _project_inputs).
    # The training dataset.py does this via get_visibility_mask before feeding
    # examples to the model. We must replicate that here or the pose_enc Linear
    # layer (133 in) gets a 132-dim tensor and raises a matmul shape error.
    visibility_mask = get_visibility_mask(landmarks)  # (target_len, 3): [pose, left_hand, right_hand]
    landmarks = np.concatenate([landmarks, visibility_mask], axis=-1)  # (target_len, 1665)
    
    # Detailed Debug Logging
    print(f"\n==================== [AI Inference Debug] ====================")
    print(f"[AI Debug] Original frames received: {T} (~{T*0.15:.2f}s)")
    print(f"[AI Debug] Resampled frames (25 fps): {target_len}")
    total_elements = landmarks.size
    zero_elements = np.sum(landmarks[:, :1662] == 0.0)  # count zeros in landmark coords only (not mask)
    zero_percentage = (zero_elements / (target_len * 1662)) * 100
    print(f"[AI Debug] Feature shape: {landmarks.shape}")
    print(f"[AI Debug] Total zero elements (landmark coords): {zero_elements} / {target_len * 1662} ({zero_percentage:.1f}%)")
    print(f"[AI Debug] Pose detected (% frames):      {visibility_mask[:, 0].mean()*100:.1f}%")
    print(f"[AI Debug] Left hand detected (% frames):  {visibility_mask[:, 1].mean()*100:.1f}%")
    print(f"[AI Debug] Right hand detected (% frames): {visibility_mask[:, 2].mean()*100:.1f}%")
    
    # Check left hand (dims 1536-1599) and right hand (dims 1599-1662) zeros
    left_hand_zeros = np.sum(landmarks[:, 1536:1599] == 0.0) / (63 * target_len) * 100
    right_hand_zeros = np.sum(landmarks[:, 1599:1662] == 0.0) / (63 * target_len) * 100
    print(f"[AI Debug] Left hand coord zeros:  {left_hand_zeros:.1f}%")
    print(f"[AI Debug] Right hand coord zeros: {right_hand_zeros:.1f}%")
    
    landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(device) # (1, T, 1665)
    landmark_mask = torch.ones(1, landmarks.shape[0], dtype=torch.float32).to(device) # (1, T)
    
    with torch.no_grad():
        generated_ids = model.generate(
            landmarks=landmarks_tensor,
            landmark_mask=landmark_mask,
            tokenizer=tokenizer,
            num_beams=4,
            max_length=128
        )
    translation = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"[AI Debug] Generated Token IDs: {generated_ids[0].tolist()}")
    print(f"[AI Debug] Final Decoded Sentence: '{translation}'")
    
    # ── Refinement Layer (signlinkLLM architecture) ──────────────────────────
    # Strategy:
    #   1. Try Groq / Llama-3.3-70B (cloud, primary — signlinkLLM)
    #   2. Fall back to local Qwen 0.5B if Groq is unavailable
    #   3. Return raw mBART output if both fail
    # ─────────────────────────────────────────────────────────────────────────
    refined_translation = translation
    confidence = "unknown"

    # 1. Groq primary path (signlinkLLM)
    if groq_chain is not None:
        try:
            result = groq_chain.invoke({"sign_text": translation})
            refined_translation = result.corrected_sentence.strip()
            confidence = result.confidence
            print(f"[SignLinkLLM] Groq refined: '{refined_translation}' (confidence={confidence})")
        except Exception as groq_err:
            print(f"[SignLinkLLM] Groq refinement failed ({groq_err}), trying Qwen fallback...")
            # Fall through to Qwen below
            refined_translation = translation  # reset to raw mBART output

    # 2. Qwen local fallback path (only used if Groq is absent or failed)
    if refined_translation == translation and llm_model is not None and llm_tokenizer is not None:
        try:
            qwen_prompt = (
                f"<|im_start|>system\nYou are a Sign Language Translation Refinement assistant. "
                f"Convert the raw sign-recognition text into a single, natural, grammatically correct English sentence. "
                f"Preserve meaning. Do not add explanations.<|im_end|>\n"
                f"<|im_start|>user\nSign text: {translation}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            inputs = llm_tokenizer(qwen_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = llm_model.generate(
                    **inputs,
                    max_new_tokens=32,
                    pad_token_id=llm_tokenizer.eos_token_id
                )
            decoded = llm_tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()
            if decoded.startswith('"') and decoded.endswith('"'):
                decoded = decoded[1:-1]
            refined_translation = decoded
            confidence = "medium"  # Qwen is less calibrated, default to medium
            print(f"[LLM Fallback] Qwen refined: '{refined_translation}'")
        except Exception as qwen_err:
            print(f"[LLM Fallback] Qwen refinement failed: {qwen_err}")

    print(f"[AI Debug] Raw mBART:       '{translation}'")
    print(f"[AI Debug] Final output:    '{refined_translation}' (confidence={confidence})")
    print(f"=================================================================\n")
    return refined_translation

@app.on_event("shutdown")
def shutdown_models() -> None:
    hands.close()
    face_mesh.close()
    # Close any active sessions
    with sessions_lock:
        for s in active_sessions.values():
            s["holistic"].close()
        active_sessions.clear()