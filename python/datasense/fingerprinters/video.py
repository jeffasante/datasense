import cv2
import json
import sys
import numpy as np
from typing import List, Dict, Any

class VideoFingerprinter:
    """Extracts properties from video datasets."""
    
    def analyze(self, video_paths: List[str]) -> Dict[str, Any]:
        """Analyzes a list of video files."""
        results: Dict[str, Any] = {
            "fps": 0,
            "duration_stats": {"mean": 0.0},
            "resolution": [0, 0],
            "motion_density": 0.0,
            "scene_cut_rate": 0.0,
            "temporal_complexity": "low",
            "has_audio_track": False,
            "sample_count": len(video_paths)
        }
        
        all_fps = []
        all_resolutions = []
        
        for path in video_paths:
            try:
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    continue
                    
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if fps > 0:
                    all_fps.append(fps)
                    results["duration_stats"]["mean"] += frame_count / fps
                
                all_resolutions.append((width, height))
                cap.release()
            except Exception:
                continue
        
        if all_fps:
            results["fps"] = int(np.mean(all_fps))
            results["duration_stats"]["mean"] /= len(video_paths)
            
            # Median resolution
            res_array = np.array(all_resolutions)
            results["resolution"] = np.median(res_array, axis=0).tolist()
            
        return results

if __name__ == "__main__":
    import sys
    # Example usage: python video.py path1 path2 ...
    paths = sys.argv[1:]
    fingerprinter = VideoFingerprinter()
    print(json.dumps(fingerprinter.analyze(paths)))
