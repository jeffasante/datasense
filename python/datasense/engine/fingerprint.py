import os
import json
import sys
from typing import Dict, Any, List
from datasense.fingerprinters.image import ImageFingerprinter
from datasense.fingerprinters.audio import AudioFingerprinter
from datasense.fingerprinters.video import VideoFingerprinter
from datasense.fingerprinters.tabular import TabularFingerprinter
from datasense.fingerprinters.text import TextFingerprinter
from datasense.fingerprinters.mixed import MixedDatasetFingerprinter

class FingerprintEngine:
    """Orchestrates modality-specific fingerprinters into a unified dataset profile."""

    def __init__(self):
        self.printers = {
            "image": ImageFingerprinter(),
            "audio": AudioFingerprinter(),
            "video": VideoFingerprinter(),
            "tabular": TabularFingerprinter(),
            "text": TextFingerprinter()
        }
        self.mixed = MixedDatasetFingerprinter()

    def fingerprint_all(self, paths: Dict[str, List[str]]) -> Dict[str, Any]:
        """Analyzes all provided paths across modalities."""
        fingerprints = {}
        
        for modality, file_list in paths.items():
            if modality in self.printers and file_list:
                try:
                    fp = self.printers[modality].analyze(file_list)
                    fingerprints[modality] = fp
                except Exception as e:
                    print(f"Error fingerprinting {modality}: {str(e)}", file=sys.stderr)

        if fingerprints:
            # Wrap mixed data as well
            fingerprints["mixed"] = self.mixed.analyze(fingerprints)

        return fingerprints
