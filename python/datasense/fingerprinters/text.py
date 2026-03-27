import os
import re
from typing import Dict, Any, List
import numpy as np

class TextFingerprinter:
    """Extracts features from plain text datasets."""

    def analyze(self, file_paths: List[str]) -> Dict[str, Any]:
        """Analyzes a collection of text files to extract high-level signals."""
        if not file_paths:
            return {}

        lengths = []
        word_counts = []
        unique_words = set()
        total_chars = 0
        total_words = 0
        
        # Limit analysis for very large files to maintain speed
        MAX_SAMPLES = 1000
        sampled_paths = file_paths[:MAX_SAMPLES]
        
        for path in sampled_paths:
            try:
                if not os.path.exists(path): continue
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(50000)
                    lengths.append(len(content))
                    
                    # Tokenize simply for stats
                    words = re.findall(r'\w+', content.lower())
                    word_counts.append(len(words))
                    total_words += len(words)
                    total_chars += len(content)
                    
                    # Sample unique words to estimate vocabulary
                    if len(unique_words) < 5000:
                        unique_words.update(words[:500])
            except:
                continue

        if not lengths:
            return {"sample_count": 0}

        avg_len = np.mean(lengths)
        avg_word_count = np.mean(word_counts)
        
        # Density: characters per word (rough estimate of word complexity)
        complexity = total_chars / (total_words + 1)
        
        # Diversity: unique words per word
        diversity = len(unique_words) / (total_words / len(sampled_paths) + 1)

        return {
            "modality": "text",
            "sample_count": len(file_paths),
            "stats": {
                "avg_char_length": float(avg_len),
                "avg_word_count": float(avg_word_count),
                "vocab_richness": float(diversity),
                "word_complexity": float(complexity)
            },
            "signals": self._extract_signals(avg_word_count, diversity)
        }

    def _extract_signals(self, avg_words: float, diversity: float) -> List[str]:
        signals = []
        if avg_words > 500:
            signals.append("long_form_document")
        elif avg_words < 50:
            signals.append("short_form_text")
            
        if diversity > 0.6:
            signals.append("high_vocabulary_diversity")
        elif diversity < 0.2:
            signals.append("low_vocabulary_diversity")
            
        return signals
