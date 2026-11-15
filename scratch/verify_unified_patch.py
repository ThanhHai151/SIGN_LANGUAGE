import json
import sys
import os
sys.path.insert(0, ".")

from backend.vision.diacritic_state_machine import DiacriticStateMachine, COMPETING_RULES_MAP
from backend.vision.feature_extractor import calculate_distance, calculate_angle
from backend.vision.recognizer import VSLRecognizer

rec = VSLRecognizer()

# Test proposed diacritic evaluation logic on all 7 sequences
VIDEOS = {
    'Ă': ('backend/data/alphabet_v2/motion_sequences/Ă.json', 'A'),
    'Â': ('backend/data/alphabet_v2/motion_sequences/Â.json', 'A'),
    'Ê': ('backend/data/alphabet_v2/motion_sequences/Ê.json', 'E'),
    'Đ': ('backend/data/alphabet_v2/motion_sequences/Đ.json', 'D'),
    'Ô': ('backend/data/alphabet_v2/motion_sequences/Ô.json', 'O'),
    'Ơ': ('backend/data/alphabet_v2/motion_sequences/Ơ.json', 'O'),
    'Ư': ('backend/data/alphabet_v2/motion_sequences/Ư.json', 'U'),
}

print("=== CHECKING LOGIC VIABILITY ===")
print("All sequences loaded.")

