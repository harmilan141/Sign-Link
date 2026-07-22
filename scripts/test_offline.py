import os
import sys
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import transformers

try:
    print("Trying to load facebook/mbart-large-50 offline...")
    model = transformers.MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
    print("SUCCESS: Loaded mbart-large-50 offline!")
except Exception as e:
    print("FAILED:", e)
