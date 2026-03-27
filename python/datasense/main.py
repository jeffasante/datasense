import os
import sys
import json
from typing import Dict, Any, List

# Standard DataSense engine imports
from datasense.engine.fingerprint import FingerprintEngine
from datasense.engine.scoring import ScoringEngine
from datasense.engine.recommendations import RecommendationEngine
from datasense.engine.plots import generate_plots
from datasense.engine.exporter import ReportExporter
from datasense.engine.generators import TrainingGenerator
from datasense.engine.explain import ExplanationEngine
from datasense.engine.tui import DashboardTUI

def main():
    try:
        # Read from stdin
        input_raw = sys.stdin.read()
        if not input_raw:
            return
            
        input_data = json.loads(input_raw)
        action = input_data.get("action")
        
        # Determine fingerprints — either from paths or direct input
        paths = input_data.get("paths", {})
        if paths:
            engine = FingerprintEngine()
            fingerprints = engine.fingerprint_all(paths)
        else:
            fingerprints = input_data.get("fingerprints", {})
        
        if action == "score":
            scorer = ScoringEngine()
            ranked = scorer.score_all(fingerprints)
            
            result = {
                "fingerprints": fingerprints,
                "leaderboard": ranked,
                "summary": ExplanationEngine().explain(fingerprints, ranked)
            }
            
            report_path = input_data.get("report_md")
            plot_dir = input_data.get("plot_dir")
            theme = "light"
            
            if plot_dir:
                plot_paths = generate_plots(fingerprints, ranked, plot_dir, theme=theme)
                result["plots"] = plot_paths
            
            if report_path:
                exporter = ReportExporter()
                if report_path.lower().endswith(".docx"):
                    exporter.export_docx(result, report_path)
                elif report_path.lower().endswith(".pdf"):
                    exporter.export_pdf(result, report_path)
                elif report_path.lower().endswith(".txt"):
                    exporter.export_txt(result, report_path)
                else:
                    exporter.export_markdown(result, report_path)
                result["report_exported"] = report_path
            
            print(json.dumps(result))
            
        elif action == "init":
            output_blueprint = input_data.get("output_blueprint", "training_blueprint.py")
            engine = RecommendationEngine()
            recommendation = engine.recommend(fingerprints)
            
            gen = TrainingGenerator()
            gen.generate_blueprint(recommendation, output_blueprint)
            
            print(json.dumps({
                "recommendation": recommendation,
                "blueprint_saved": output_blueprint
            }))
            
        elif action == "dashboard":
            scorer = ScoringEngine()
            ranked = scorer.score_all(fingerprints)
            result = {
                "fingerprints": fingerprints,
                "leaderboard": ranked,
                "summary": ExplanationEngine().explain(fingerprints, ranked)
            }
            tui = DashboardTUI()
            tui.render(result)
            sys.exit(0)
            
        else:  # analyze
            engine = RecommendationEngine()
            recommendation = engine.recommend(fingerprints)
            ranked = ScoringEngine().score_all(fingerprints)
            
            result = {
                "fingerprints": fingerprints,
                "recommendation": recommendation,
                "leaderboard": ranked,
                "summary": ExplanationEngine().explain(fingerprints, ranked)
            }
            
            report_path = input_data.get("report_md")
            plot_dir = input_data.get("plot_dir")
            theme = "light"
            
            if plot_dir:
                plot_paths = generate_plots(fingerprints, ranked, plot_dir, theme=theme)
                result["plots"] = plot_paths

            if report_path:
                exporter = ReportExporter()
                if report_path.lower().endswith(".docx"):
                    exporter.export_docx(result, report_path)
                elif report_path.lower().endswith(".pdf"):
                    exporter.export_pdf(result, report_path)
                elif report_path.lower().endswith(".txt"):
                    exporter.export_txt(result, report_path)
                else:
                    exporter.export_markdown(result, report_path)
                result["report_exported"] = report_path

            print(json.dumps(result))
        
    except Exception as e:
        import traceback
        # print(traceback.format_exc(), file=sys.stderr)
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()
