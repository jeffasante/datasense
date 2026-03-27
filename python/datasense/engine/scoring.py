import json
from typing import Dict, Any, List

# Registry of candidate models per modality
CANDIDATES = {
    "image": [
        "EfficientNet-B0", "ResNet-50", "MobileNetV3",
        "Vision Transformer (ViT)", "Swin Transformer",
        "ConvNeXt", "DeiT-Small"
    ],
    "audio": [
        "Wav2Vec2", "Whisper", "HuBERT",
        "LSTM-RNN", "ECAPA-TDNN", "AST"
    ],
    "tabular": [
        "XGBoost", "LightGBM", "CatBoost",
        "TabNet", "FT-Transformer", "Random Forest",
        "Logistic Regression", "MLP"
    ],
    "video": [
        "TimeSformer", "SlowFast", "3D CNN (R3D-101)",
        "C3D", "X3D", "VideoMAE"
    ],
    "text": [
        "BERT / DistilBERT", "GPT-2 / Llama-3", "FastText",
        "Bi-LSTM", "TF-IDF + Ridge Classifier", "CNN-Text"
    ],
    "mixed": [
        "CLIP", "Flamingo", "Custom Fusion MLP",
        "ImageBind", "Perceiver IO", "Whisper (ASR)", "VITS (TTS)", "PPO (RL)"
    ]
}

class ScoringEngine:
    """Calculates how well a specific architecture fits a given dataset fingerprint."""
    
    def score_all(self, fingerprint: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scores ALL relevant candidate models and returns a ranked list."""
        detected = []
        for mod in ["mixed", "video", "image", "audio", "tabular", "text"]:
            if mod in fingerprint:
                detected.append(mod)
        
        # Collect candidate models from all detected modalities
        candidates = []
        seen = set()
        for mod in detected:
            for model in CANDIDATES.get(mod, []):
                if model not in seen:
                    candidates.append(model)
                    seen.add(model)
        
        # Score each candidate
        results = []
        for model in candidates:
            result = self.score(fingerprint, model)
            results.append(result)
        
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def score(self, fingerprint: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """Calculates a fit score (0.0 - 1.0) and provides justification."""
        model_key = model_name.lower()
        
        result = {
            "model": model_name,
            "score": 0.0,
            "justification": "No specific profile for this model-data combination."
        }
        
        # Try all present modalities and keep the best score
        if "mixed" in fingerprint:
            self._score_mixed(fingerprint["mixed"], model_key, result)
        
        # Try each modality — keep the highest score
        best = dict(result)
        for mod_key, scorer_fn in [
            ("audio", self._score_audio),
            ("tabular", self._score_tabular),
            ("text", self._score_text),
        ]:
            if mod_key in fingerprint:
                candidate = {"model": model_name, "score": 0.0, "justification": result["justification"]}
                scorer_fn(fingerprint[mod_key], model_key, candidate)
                if float(candidate["score"]) > float(best["score"]):
                    best = candidate
        
        # Merge best into result
        if float(best["score"]) > float(result["score"]):
            result["score"] = float(best["score"])
            result["justification"] = str(best["justification"])
            
        return result


    def _score_mixed(self, fp: Dict[str, Any], model: str, result: Dict[str, Any]):
        modalities = fp.get("modalities", [])
        if "clip" in model and "image" in modalities:
            result["score"] = 0.95
            result["justification"] = "Industry standard for image-text alignment and zero-shot tasks."
        elif "flamingo" in model:
            result["score"] = 0.9
            result["justification"] = "Powerful multi-modal LLM for visual question answering."
        elif "imagebind" in model:
            result["score"] = 0.88
            result["justification"] = "Binds multiple modalities into a unified embedding space."
        elif "perceiver" in model:
            result["score"] = 0.82
            result["justification"] = "Flexible architecture that generalizes across any modality."
        elif "fusion" in model:
            result["score"] = 0.85
            result["justification"] = "Custom fusion approach allows tailored cross-modal interaction."

    def _score_video(self, fp: Dict[str, Any], model: str, result: Dict[str, Any]):
        if "timesformer" in model:
            result["score"] = 0.92
            result["justification"] = "Divided space-time attention for complex video understanding."
        elif "slowfast" in model:
            result["score"] = 0.9
            result["justification"] = "Dual-pathway design captures both slow and fast motion."
        elif "x3d" in model:
            result["score"] = 0.88
            result["justification"] = "Efficient progressive expansion across spatial, temporal, width, and depth."
        elif "videomae" in model:
            result["score"] = 0.87
            result["justification"] = "Self-supervised pretraining excels with limited labeled video data."
        elif "c3d" in model:
            result["score"] = 0.8
            result["justification"] = "Lightweight 3D convolutions, good for feature extraction."
        elif "3d" in model or "r3d" in model:
            result["score"] = 0.85
            result["justification"] = "Classical 3D convolutions for short-range temporal patterns."


    def _score_image(self, fp: Dict[str, Any], model: str, result: Dict[str, Any]):
        res = fp.get("resolution", {}).get("median", [224, 224])
        samples = fp.get("sample_count", 0)
        
        if "efficientnet" in model:
            result["score"] = 0.85
            result["justification"] = "Excellent accuracy-efficiency balance across resolutions."
        elif "resnet" in model:
            result["score"] = 0.75
            result["justification"] = "Proven baseline for spatial patterns."
            if res[0] > 1024:
                result["score"] -= 0.1
                result["justification"] += " May struggle with very high resolution."
        elif "mobilenet" in model:
            result["score"] = 0.72
            result["justification"] = "Optimized for edge/mobile deployment with acceptable accuracy."
        elif "vit" in model or "deit" in model:
            if samples > 10000:
                result["score"] = 0.9
                result["justification"] = "Transformers excel at large datasets with global context."
            else:
                result["score"] = 0.4
                result["justification"] = "Small text dataset; recommend lightweight models like FastText."
        elif "swin" in model:
            result["score"] = 0.88
            result["justification"] = "Shifted-window attention captures multi-scale spatial features."
        elif "convnext" in model:
            result["score"] = 0.83
            result["justification"] = "Modern CNN matching transformer performance with simpler design."

    def _score_audio(self, fp: Dict[str, Any], model: str, result: Dict[str, Any]):
        dur = fp.get("duration_stats", {}).get("mean", 0.0)
        
        if "whisper" in model:
            if dur > 5.0:
                result["score"] = 0.95
                result["justification"] = "Strong candidate for long-form speech and translation."
            else:
                result["score"] = 0.7
                result["justification"] = "Good, but potentially overkill for short clips."
        elif "wav2vec" in model:
            result["score"] = 0.88
            result["justification"] = "Ideal for low-resource or short temporal dependencies."
        elif "hubert" in model:
            result["score"] = 0.86
            result["justification"] = "Strong self-supervised features for speech representation."
        elif "lstm" in model or "rnn" in model:
            result["score"] = 0.65
            result["justification"] = "Classic sequential model; outperformed by modern transformers."
        elif "ecapa" in model or "tdnn" in model:
            result["score"] = 0.84
            result["justification"] = "State-of-the-art for speaker verification and embeddings."
        elif "ast" in model:
            result["score"] = 0.82
            result["justification"] = "Audio Spectrogram Transformer; treats audio as visual patches."

    def _score_tabular(self, fp: Dict[str, Any], model: str, result: Dict[str, Any]):
        missing = fp.get("missing_rate", 0.0)
        
        if "xgboost" in model:
            result["score"] = 0.9
            result["justification"] = "Gradient boosting with native missing value handling."
            if missing < 0.01:
                result["score"] = 0.95
        elif "lightgbm" in model or "lgbm" in model:
            result["score"] = 0.92
            result["justification"] = "Fast gradient boosting with histogram-based splits."
        elif "catboost" in model:
            result["score"] = 0.89
            result["justification"] = "Excellent handling of categorical features without encoding."
        elif "tabnet" in model:
            result["score"] = 0.7
            result["justification"] = "Attention-based deep learning for tabular, requires tuning."
        elif "ft-transformer" in model or "ft_transformer" in model:
            result["score"] = 0.72
            result["justification"] = "Feature tokenizer + transformer; competitive but complex."
        elif "random forest" in model or "rf" in model:
            result["score"] = 0.78
            result["justification"] = "Robust ensemble baseline with minimal tuning needed."
        elif "logistic" in model:
            result["score"] = 0.55
            result["justification"] = "Simple linear model; only suited for linearly separable data."
        elif b"mlp" in model.encode() or "mlp" in model:
            result["score"] = 0.6
            result["justification"] = "Basic neural network; usually outperformed by tree-based methods."

    def _score_text(self, fp: Dict[str, Any], model: str, result: Dict[str, Any]):
        stats = fp.get("stats", {})
        avg_words = stats.get("avg_word_count", 0)
        diversity = stats.get("vocab_richness", 0)
        
        if "bert" in model or "distilbert" in model:
            if avg_words < 512:
                result["score"] = 0.94
                result["justification"] = "Transformer-based encoder; ideal for understanding and classification tasks."
            else:
                result["score"] = 0.7
                result["justification"] = "Strong but limited by context window of 512 tokens."
        elif "gpt" in model or "llama" in model:
            result["score"] = 0.92
            result["justification"] = "Causal transformer excels at generative tasks and understanding global context."
        elif "fasttext" in model:
            result["score"] = 0.82
            result["justification"] = "Fast, embedding-based classifier that handles rare words well."
        elif "lstm" in model:
            result["score"] = 0.68
            result["justification"] = "Recurrent network; good for sequence, but outperformed by transformers."
        elif "tf-idf" in model or "tfidf" in model:
            result["score"] = 0.75
            if diversity < 0.3:
                result["score"] += 0.1
                result["justification"] = "Highly effective for low-diversity datasets with clear keywords."
            else:
                result["justification"] = "Solid baseline for simple classification tasks."
        elif "cnn" in model:
            result["score"] = 0.74
            result["justification"] = "Efficient for finding local n-gram patterns in text."

    def _score_asr(self, fp: Dict[str, Any], model: str, result: Dict[str, Any]):
        """Scores ASR model suitability."""
        result["score"] = 0.8
        result["justification"] = "Good audio-text alignment detected for speech recognition."

    def _score_tts(self, fp: Dict[str, Any], model: str, result: Dict[str, Any]):
        """Scores TTS model suitability."""
        result["score"] = 0.75
        result["justification"] = "High quality text-speech pairs for synthesis tasks."

    def _score_rl(self, fp: Dict[str, Any], model: str, result: Dict[str, Any]):
        """Scores RL model suitability."""
        result["score"] = 0.7
        result["justification"] = "Sequential decision making and environment states detected."

if __name__ == "__main__":
    engine = ScoringEngine()
    dummy_fp = {"image": {"resolution": {"median": [1024, 1024]}, "sample_count": 500}}
    for r in engine.score_all(dummy_fp):
        print(f"  {r['model']:25s} → {int(r['score']*100)}% — {r['justification']}")
