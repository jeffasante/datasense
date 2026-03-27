import json
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple

class ImageFingerprinter:
    """Extracts properties from image datasets."""
    
    def analyze(self, image_paths: List[str]) -> Dict[str, Any]:
        """Analyzes a list of image paths."""
        results: Dict[str, Any] = {
            "resolution": {"min": [0, 0], "max": [0, 0], "median": [0, 0]},
            "color_channels": 3,
            "sample_count": len(image_paths),
            "spatial_complexity": 0.0,
            "class_count": 0,
            "class_balance": 1.0,
        }
        
        resolutions: List[Tuple[int, int]] = []
        for path in image_paths:
            try:
                with Image.open(path) as img:
                    w, h = img.size
                    resolutions.append((w, h))
                    
                    # Basic channel detection from first image
                    if results["color_channels"] == 3:
                        results["color_channels"] = len(img.getbands())
            except Exception:
                continue
        
        if resolutions:
            res_array = np.array(resolutions)
            min_res = res_array.min(axis=0).tolist()
            max_res = res_array.max(axis=0).tolist()
            med_res = np.median(res_array, axis=0).tolist()
            
            # Use calm comments for logic clarity
            results["resolution"]["min"] = min_res
            results["resolution"]["max"] = max_res
            results["resolution"]["median"] = med_res
            
        return results

if __name__ == "__main__":
    import sys
    # Example usage: python image.py path1 path2 ...
    paths = sys.argv[1:]
    fingerprinter = ImageFingerprinter()
    print(json.dumps(fingerprinter.analyze(paths)))
