import json
from typing import List, Dict, Any

class MixedDatasetFingerprinter:
    """Combines per-modality fingerprints and adds fusion analysis."""
    
    def analyze(self, fingerprints: Dict[str, Any]) -> Dict[str, Any]:
        """Analyzes a collection of per-modality fingerprints."""
        modalities = list(fingerprints.keys())
        
        results: Dict[str, Any] = {
            "modalities": modalities,
            "alignment": "unpaired",
            "dominant_modality": "unknown",
            "fusion_complexity": "late"
        }
        
        if not modalities:
            return results
            
        # Basic dominance heuristic
        sample_counts = {m: fingerprints[m].get("sample_count", 0) for m in modalities}
        if sample_counts:
            # Using lambda for linter clarity
            results["dominant_modality"] = max(sample_counts, key=lambda k: sample_counts[k])
            
        # Alignment heuristic (simple check if counts match)
        counts = list(sample_counts.values())
        if all(c == counts[0] for c in counts) and len(counts) > 1:
            results["alignment"] = "paired"
        elif any(c > 0 for c in counts) and len(counts) > 1:
            results["alignment"] = "partial"
            
        # Fusion complexity heuristic
        if "tabular" in modalities and ("image" in modalities or "audio" in modalities):
            results["fusion_complexity"] = "hybrid"
            
        return results

if __name__ == "__main__":
    import sys
    # Expects input JSON from stdin
    try:
        input_data = json.load(sys.stdin)
        fingerprinter = MixedDatasetFingerprinter()
        print(json.dumps(fingerprinter.analyze(input_data)))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
