import json
import os
import sys
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Ensure .env is loaded
load_dotenv()

try:
    from groq import Groq
    from openai import OpenAI
    from anthropic import Anthropic
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

class ExplanationEngine:
    """Advanced AI Reasoning Engine with Multimodal and Tool Support."""

    def __init__(self):
        self.provider = os.getenv("AI_PROVIDER", "groq").lower()
        self.api_key = os.getenv("AI_API_KEY") or os.getenv("GROQ_API_KEY")
        self.model = os.getenv("AI_MODEL")
        self.client = None
        
        if SDK_AVAILABLE and self.api_key:
            self._initialize_client()

    def _initialize_client(self):
        """Standardized client initialization across major providers."""
        try:
            if not self.api_key: return

            if self.provider == "groq":
                self.client = Groq(api_key=self.api_key)
                if not self.model: self.model = "groq/compound-mini"
            elif self.provider in ["openai", "grok", "moonshot"]:
                base_url = None
                if self.provider == "grok": base_url = "https://api.x.ai/v1"
                if self.provider == "moonshot": base_url = "https://api.moonshot.cn/v1"
                
                self.client = OpenAI(api_key=self.api_key, base_url=base_url)
                if not self.model: 
                    self.model = "gpt-4o" if self.provider == "openai" else "grok-2"
            elif self.provider == "anthropic":
                self.client = Anthropic(api_key=self.api_key)
                if not self.model: self.model = "claude-3-5-sonnet-latest"
        except Exception:
            self.client = None

    def explain(self, fingerprints: Dict[str, Any], ranked: List[Dict[str, Any]]) -> str:
        """Deep architectural summary powered by high-performance LLMs."""
        if not ranked:
            return "No architectures were ranked."

        if self.client:
            try:
                prompt = self._get_ai_prompt(fingerprints, ranked)
                return self._call_ai(prompt)
            except Exception as e:
                # Log error for transparency
                sys.stderr.write(f"\n[DataSense AI] Connection Error: {str(e)}\n")

        return self._rule_based_explain(fingerprints, ranked)

    def _call_ai(self, prompt: str) -> str:
        """Dispatches call with support for advanced tool-enabled models."""
        if not self.client: return ""
        
        role_msg = "You are DataSense AI, an expert machine learning architect. " \
                   "Use your tools to research SOTA architectures if the current recommendations need deeper validation."
        
        if self.provider == "groq":
            # Advanced Compound Model Support
            full_content = ""
            params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": role_msg},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 1,
                "max_completion_tokens": 2048,
                "top_p": 1,
                "stream": True,
                "stop": None
            }
            
            # Special handling for groq/compound models
            if "compound" in self.model:
                params["compound_custom"] = {
                    "tools": {
                        "enabled_tools": ["web_search", "code_interpreter", "visit_website"]
                    }
                }
            
            completion = self.client.chat.completions.create(**params)
            
            for chunk in completion:
                content = chunk.choices[0].delta.content or ""
                full_content += content
            return full_content

        elif self.provider in ["openai", "grok", "moonshot"]:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": role_msg},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.3
            )
            return completion.choices[0].message.content
        
        elif self.provider == "anthropic":
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=role_msg,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        
        return ""

    def _rule_based_explain(self, fingerprints: Dict[str, Any], ranked: List[Dict[str, Any]]) -> str:
        """Fallback rule-based logic."""
        primary = ranked[0]
        narrative = f"### Executive Summary (Strategic Recommendation)\n\n"
        narrative += f"Based on the multimodal analysis, **{primary['model']}** is the optimal choice for this dataset. "
        narrative += f"The justification provided by the scoring engine is as follows: {primary['justification']}\n"
        return narrative

    def _get_ai_prompt(self, fingerprints: Dict[str, Any], ranked: List[Dict[str, Any]]) -> str:
        """Context-rich prompt for deep architectural reasoning."""
        context = {"fingerprints": fingerprints, "leaderboard": ranked[:5]}
        return f"Provide a comprehensive, professional executive summary for the following architectural analysis. " \
               f"Be specific about how the detected modalities interact and why the top-ranked model is the SOTA choice. " \
               f"JSON CONTEXT: {json.dumps(context)}"
