# prosody.py — force import from local prosody3 package (hardcoded path)

import sys
_PROSODY3_ROOT = r"C:\Users\trist\OneDrive\Documents\Remastered TTS Final Version\TTS Core Remastered\prosody3"
if _PROSODY3_ROOT not in sys.path:
    sys.path.insert(0, _PROSODY3_ROOT)

from prosody3.prosody_predictor import ProsodyPredictorV15  # noqa: E402

has_prosody3 = True

def run(context):
    return {"prosody_status": "ok", "source": "prosody3"}
