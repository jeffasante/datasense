import pandas as pd
import numpy as np
import json
import sys
from typing import List, Dict, Any

class TabularFingerprinter:
    """Extracts properties from tabular datasets."""
    
    def analyze(self, tabular_paths: List[str]) -> Dict[str, Any]:
        """Analyzes a list of tabular datasets (CSV, Parquet, JSONL)."""
        results: Dict[str, Any] = {
            "row_count": 0,
            "feature_count": 0,
            "missing_rate": 0.0,
            "cardinality": {"low": 0, "high": 0, "unique": 0},
            "feature_types": {"numeric": 0, "categorical": 0, "datetime": 0},
            "correlation_density": 0.0,
            "sample_count": len(tabular_paths)
        }
        
        for path in tabular_paths:
            try:
                # Basic tabular loading support
                if path.endswith(".csv"):
                    df = pd.read_csv(path)
                elif path.endswith(".parquet"):
                    df = pd.read_parquet(path)
                elif path.endswith(".jsonl"):
                    df = pd.read_json(path, lines=True)
                else:
                    continue
                
                results["row_count"] += len(df)
                results["feature_count"] = len(df.columns)
                results["missing_rate"] = float(df.isnull().mean().mean())
                
                # Determine feature types
                results["feature_types"]["numeric"] = len(df.select_dtypes(include=[np.number]).columns)
                results["feature_types"]["categorical"] = len(df.select_dtypes(include=["object", "category"]).columns)
                results["feature_types"]["datetime"] = len(df.select_dtypes(include=["datetime"]).columns)
                
                # Simple cardinality analysis
                threshold_high = 100
                threshold_low = 10
                
                for col in df.columns:
                    nunique = df[col].nunique()
                    if nunique >= len(df) * 0.9:
                        results["cardinality"]["unique"] += 1
                    elif nunique > threshold_high:
                        results["cardinality"]["high"] += 1
                    elif nunique < threshold_low:
                        results["cardinality"]["low"] += 1
                
                # Simple correlation density (for numeric columns)
                numeric_df = df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) > 1:
                    corr = numeric_df.corr().abs()
                    avg_corr = (corr.sum().sum() - len(numeric_df.columns)) / (len(numeric_df.columns)**2 - len(numeric_df.columns))
                    results["correlation_density"] = float(avg_corr)
                    
            except Exception:
                continue
                
        return results

if __name__ == "__main__":
    paths = sys.argv[1:]
    fingerprinter = TabularFingerprinter()
    print(json.dumps(fingerprinter.analyze(paths)))
