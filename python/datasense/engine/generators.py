import os
from typing import Dict, Any

class TrainingGenerator:
    """Generates starter training code and configurations based on recommendations."""

    def generate_blueprint(self, recommendation: Dict[str, Any], output_path: str):
        """Creates a standalone Python training script for the recommended architecture."""
        model = recommendation.get("primary", "Model")
        
        # Simple templates per model family
        templates = {
            "CLIP": self._clip_template(),
            "XGBoost": self._xgboost_template(),
            "BERT": self._bert_template(),
            "TimeSformer": self._timesformer_template(),
            "EfficientNet": self._efficientnet_template()
        }
        
        # Find best matching template
        content = templates.get("EfficientNet") # Default
        for key in templates:
            if key.lower() in model.lower():
                content = templates[key]
                break
        
        with open(output_path, 'w') as f:
            f.write(content.replace("{{MODEL_NAME}}", model))

    def _clip_template(self) -> str:
        return """
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# DataSense Blueprint for {{MODEL_NAME}}
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def train_step(image_path, text):
    inputs = processor(text=[text], images=[Image.open(image_path)], return_tensors="pt", padding=True)
    outputs = model(**inputs)
    return outputs.logits_per_image
"""

    def _xgboost_template(self) -> str:
        return """
import xgboost as xgb
import pandas as pd

# DataSense Blueprint for {{MODEL_NAME}}
def train(csv_path, target_col):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    dtrain = xgb.DMatrix(X, label=y)
    param = {'max_depth': 6, 'eta': 0.3, 'objective': 'binary:logistic'}
    num_round = 100
    bst = xgb.train(param, dtrain, num_round)
    return bst
"""

    def _bert_template(self) -> str:
        return """
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# DataSense Blueprint for {{MODEL_NAME}}
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

def prepare_data(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
"""

    def _timesformer_template(self) -> str:
        return """
import torch
# DataSense Blueprint for {{MODEL_NAME}}
# Requires 'timesformer' package
# from timesformer.models.vit import TimeSformer

def init_model():
    # model = TimeSformer(img_size=224, num_frames=8, num_classes=400)
    return "TimeSformer initialized (requires package)"
"""

    def _efficientnet_template(self) -> str:
        return """
import torch
import torchvision.models as models

# DataSense Blueprint for {{MODEL_NAME}}
model = models.efficientnet_b0(pretrained=True)
model.eval()

def infer(batch):
    with torch.no_grad():
        return model(batch)
"""
