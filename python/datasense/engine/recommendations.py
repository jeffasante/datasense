import json
import sys

class RecommendationEngine:
    """Maps dataset fingerprints to suggested ML architecture families."""
    
    def __init__(self):
        # Calm comments for logic definitions
        self.rules = []

    def recommend(self, fingerprint):
        """Generates recommendations based on the merged fingerprint."""
        recommendations = {
            "primary": "XGBoost",
            "confidence": 0.5,
            "alternatives": [],
            "key_signals": []
        }
        
        # Determine modality and apply rules
        if "mixed" in fingerprint:
            self._recommend_mixed(fingerprint, recommendations)
        elif "video" in fingerprint:
            self._recommend_video(fingerprint["video"], recommendations)
        elif "image" in fingerprint:
            self._recommend_image(fingerprint["image"], recommendations)
        elif "audio" in fingerprint:
            self._recommend_audio(fingerprint["audio"], recommendations)
        elif "tabular" in fingerprint:
            self._recommend_tabular(fingerprint["tabular"], recommendations)
        elif "text" in fingerprint:
            self._recommend_text(fingerprint["text"], recommendations)
            
        # Detect Reinforcement Learning tasks (e.g. high sequential tabular)
        if "tabular" in fingerprint:
            self._detect_rl(fingerprint["tabular"], recommendations)
            
        return recommendations

    def _recommend_mixed(self, fp, recs):
        """Rule-based logic for multimodal datasets."""
        modalities = fp.get("mixed", {}).get("modalities", [])
        
        if "image" in modalities and "tabular" in modalities:
            recs["primary"] = "Custom Fusion MLP"
            recs["confidence"] = 0.85
            recs["key_signals"].append("tabular_image_fusion")
            recs["alternatives"] = ["CLIP", "Flamingo"]
        elif "audio" in modalities and "text" in modalities:
            recs["primary"] = "Whisper (ASR)"
            recs["confidence"] = 0.95
            recs["key_signals"].append("audio_text_asr_alignment")
            recs["alternatives"] = ["Conformer", "Wav2Vec2"]
        elif "image" in modalities and "text" in modalities:
            recs["primary"] = "CLIP"
            recs["confidence"] = 0.92
            recs["key_signals"].append("image_text_alignment")
        else:
            recs["primary"] = "Flamingo"
            recs["confidence"] = 0.8

    def _recommend_video(self, vid_fp, recs):
        """Rule-based logic for video architectures."""
        dur = vid_fp.get("duration_stats", {}).get("mean", 0.0)
        
        if dur > 30.0:
            recs["primary"] = "TimeSformer"
            recs["confidence"] = 0.88
            recs["key_signals"].append("long_sequence_spatiotemporal")
        else:
            recs["primary"] = "3D CNN (R3D-101)"
            recs["confidence"] = 0.85
            recs["key_signals"].append("short_action_clip")
            
        recs["alternatives"] = ["SlowFast", "C3D"]

    def _recommend_image(self, img_fp, recs):
        """Rule-based logic for image architectures."""
        res = img_fp.get("resolution", {}).get("median", [224, 224])
        samples = img_fp.get("sample_count", 0)
        
        if res[0] >= 1024:
            recs["primary"] = "Swin Transformer"
            recs["confidence"] = 0.85
            recs["key_signals"].append("high_res_spatial_patterns")
        elif samples > 50000:
            recs["primary"] = "Vision Transformer (ViT)"
            recs["confidence"] = 0.9
            recs["key_signals"].append("large_data_global_context")
        else:
            recs["primary"] = "EfficientNet-B0"
            recs["confidence"] = 0.82
            recs["key_signals"].append("small_to_medium_image_dataset")
            
        recs["alternatives"] = ["ResNet-50", "MobileNetV3"]

    def _recommend_audio(self, audio_fp, recs):
        """Rule-based logic for audio architectures."""
        dur = audio_fp.get("duration_stats", {}).get("mean", 0.0)
        
        if dur > 10.0:
            recs["primary"] = "Whisper"
            recs["confidence"] = 0.92
            recs["key_signals"].append("long_sequence_speech")
        else:
            recs["primary"] = "Wav2Vec2"
            recs["confidence"] = 0.88
            recs["key_signals"].append("high_temporal_dependency")
            
        recs["alternatives"] = ["HuBERT", "LSTM-RNN"]

    def _recommend_tabular(self, tab_fp, recs):
        """Rule-based logic for tabular architectures."""
        missing = tab_fp.get("missing_rate", 0.0)
        card = tab_fp.get("cardinality", {}).get("high", 0)
        
        if missing > 0.15:
            recs["primary"] = "XGBoost"
            recs["confidence"] = 0.85
            recs["key_signals"].append("robust_to_missing_values")
        elif card > 10:
            recs["primary"] = "TabNet"
            recs["confidence"] = 0.8
            recs["key_signals"].append("high_cardinality_categorical")
        else:
            recs["primary"] = "LightGBM"
            recs["confidence"] = 0.88
            recs["key_signals"].append("dense_tabular_efficiency")
            
        recs["alternatives"] = ["FT-Transformer", "CatBoost"]

    def _recommend_text(self, text_fp, recs):
        """Rule-based logic for text architectures."""
        stats = text_fp.get("stats", {})
        avg_words = stats.get("avg_word_count", 0)
        diversity = stats.get("vocab_richness", 0)
        
        if avg_words > 500:
            recs["primary"] = "GPT-2 / Llama-3"
            recs["confidence"] = 0.9
            recs["key_signals"].append("long_form_context")
            recs["alternatives"] = ["Longformer", "Llama-2"]
        elif diversity > 0.7:
            recs["primary"] = "BERT / DistilBERT"
            recs["confidence"] = 0.94
            recs["key_signals"].append("high_vocab_diversity")
            recs["alternatives"] = ["RoBERTa", "FastText"]
        else:
            recs["primary"] = "FastText"
            recs["confidence"] = 0.85
            recs["key_signals"].append("small_vocab_efficiency")
            recs["alternatives"] = ["Bi-LSTM", "CNN-Text"]

    def _detect_rl(self, tab_fp, recs):
        """Heuristic to detect RL suitability in tabular data."""
        # RL is often signaled by very high feature counts or specific sequential markers
        if tab_fp.get("missing_rate", 0) > 0.5: # Extreme sparsity can imply state-space search
             recs["alternatives"].append("PPO (RL)")

    def _recommend_asr(self, asr_fp, recs): # Kept as helper for mixed
        recs["primary"] = "Whisper"
        recs["confidence"] = 0.92
        recs["key_signals"].append("speech_recognition_optimized")
        recs["alternatives"] = ["Conformer", "Wav2Vec2"]

if __name__ == "__main__":
    # Expecting input JSON from stdin
    try:
        input_data = json.load(sys.stdin)
        engine = RecommendationEngine()
        result = engine.recommend(input_data)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
