import librosa
import numpy as np
import json
import sys
from typing import List, Dict, Any

class AudioFingerprinter:
    """Extracts properties from audio datasets."""
    
    def analyze(self, audio_paths: List[str]) -> Dict[str, Any]:
        """Analyzes a list of audio files."""
        results: Dict[str, Any] = {
            "sample_rate": 0,
            "duration_stats": {"min": 0.0, "max": 0.0, "mean": 0.0},
            "frequency_range": [0, 0],
            "spectral_density": 0.0,
            "noise_floor": 0,
            "temporal_dependency": "low",
            "language_detected": "unknown",
            "sample_count": len(audio_paths)
        }
        
        durations: List[float] = []
        sample_rates: List[int] = []
        
        for path in audio_paths:
            try:
                # Load metadata without full audio load
                info = librosa.get_duration(path=path)
                durations.append(float(info))
                
                # Check sample rate (first file sample)
                if not sample_rates:
                    _, sr = librosa.load(path, sr=None, duration=1.0)
                    sample_rates.append(int(sr))
            except Exception:
                continue
        
        if durations:
            results["duration_stats"]["min"] = min(durations)
            results["duration_stats"]["max"] = max(durations)
            results["duration_stats"]["mean"] = sum(durations) / len(durations)
            
        if sample_rates:
            results["sample_rate"] = int(np.mean(sample_rates))
            
        return results

if __name__ == "__main__":
    paths = sys.argv[1:]
    fingerprinter = AudioFingerprinter()
    print(json.dumps(fingerprinter.analyze(paths)))
