# Shim: route modules.prosody.prosody_predictor -> local prosody3 package

import sys
_ROOT = r"C:\Users\trist\OneDrive\Documents\Remastered TTS Final Version\TTS Core Remastered"
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from prosody3.prosody_predictor import ProsodyPredictorV15
__all__ = ["ProsodyPredictorV15"]
