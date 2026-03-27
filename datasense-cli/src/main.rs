use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use walkdir::WalkDir;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Analyze a dataset and recommend an architecture
    Analyze {
        paths: Vec<PathBuf>,
        /// Output path for a professional report (pdf, docx, txt, or markdown)
        #[arg(long)]
        report: Option<PathBuf>,
        /// Fusion strategy (early, late, auto)
        #[arg(long, default_value = "auto")]
        fusion: String,
        /// Generate visualization plots
        #[arg(long)]
        plot: bool,
    },
    /// Compare two datasets
    Compare {
        path1: PathBuf,
        path2: PathBuf,
    },
    /// Score a specific architecture against a dataset
    Score {
        path: PathBuf,
        /// Specific model to score (e.g. EfficientNet-B0)
        #[arg(long)]
        model: Option<String>,
        /// Generate visualization plots
        #[arg(long)]
        plot: bool,
        /// Output path for a professional report
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// Generate a training blueprint for a dataset
    Init {
        path: PathBuf,
        /// Output path for the training script
        #[arg(long, default_value = "training_blueprint.py")]
        output: PathBuf,
    },
    /// Interactive terminal dashboard
    Dashboard {
        path: PathBuf,
    },
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Hash, Clone, Copy)]
enum Modality {
    Image,
    Audio,
    Tabular,
    Video,
    Text,
    Unknown,
}

struct Ingestor {
    files: HashMap<Modality, Vec<PathBuf>>,
}

impl Ingestor {
    fn new() -> Self {
        Self {
            files: HashMap::new(),
        }
    }

    fn scan(&mut self, root: &Path) -> Result<()> {
        for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
            if entry.file_type().is_file() {
                let path = entry.path().to_path_buf();
                let modality = self.classify(&path);
                if modality != Modality::Unknown {
                    self.files.entry(modality).or_default().push(path);
                }
            }
        }
        Ok(())
    }

    fn classify(&self, path: &Path) -> Modality {
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
        match ext.as_str() {
            "png" | "jpg" | "jpeg" | "webp" | "bmp" => Modality::Image,
            "wav" | "mp3" | "flac" | "ogg" => Modality::Audio,
            "csv" | "parquet" | "jsonl" => Modality::Tabular,
            "mp4" | "avi" | "mov" | "mkv" => Modality::Video,
            "txt" | "md" | "json" => Modality::Text,
            _ => Modality::Unknown,
        }
    }
}

fn run_python(input_map: &serde_json::Map<String, serde_json::Value>) -> Result<serde_json::Value> {
    let input_json = serde_json::to_string(input_map)?;
    
    // Find python dir
    let mut base_dir = std::env::current_dir()?;
    while !base_dir.join("python/datasense/main.py").exists() {
        if let Some(parent) = base_dir.parent() {
            base_dir = parent.to_path_buf();
        } else {
            anyhow::bail!("Could not find python/datasense/main.py from {:?}", std::env::current_dir()?);
        }
    }
    
    let python_path = base_dir.join("python/datasense/main.py");
    
    let mut child = Command::new("python3")
        .env("PYTHONPATH", base_dir.join("python"))
        .arg(python_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let mut stdin = child.stdin.take().context("Failed to take stdin")?;
    stdin.write_all(input_json.as_bytes())?;
    drop(stdin);

    let output = child.wait_with_output()?;
    if !output.status.success() {
        anyhow::bail!("Python failed: {}", String::from_utf8_lossy(&output.stderr));
    }

    let result: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    Ok(result)
}

fn run_python_raw(input_map: &serde_json::Map<String, serde_json::Value>) -> Result<()> {
    let input_json = serde_json::to_string(input_map)?;
    
    // Find python dir
    let mut base_dir = std::env::current_dir()?;
    while !base_dir.join("python/datasense/main.py").exists() {
        if let Some(parent) = base_dir.parent() {
            base_dir = parent.to_path_buf();
        } else {
            anyhow::bail!("Could not find python/datasense/main.py from {:?}", std::env::current_dir()?);
        }
    }
    
    let python_path = base_dir.join("python/datasense/main.py");
    
    let mut child = Command::new("python3")
        .env("PYTHONPATH", base_dir.join("python"))
        .arg(python_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::inherit()) // Redirect directly to terminal
        .stderr(Stdio::inherit())
        .spawn()?;

    let mut stdin = child.stdin.take().context("Failed to take stdin")?;
    stdin.write_all(input_json.as_bytes())?;
    drop(stdin);

    let status = child.wait()?;
    if !status.success() {
        anyhow::bail!("Python failed with status {}", status);
    }
    Ok(())
}

fn build_paths_input(ingestor: &Ingestor) -> Result<serde_json::Value> {
    let mut paths_map = serde_json::Map::new();
    for (modality, files) in &ingestor.files {
        let key = match modality {
            Modality::Image => "image",
            Modality::Audio => "audio",
            Modality::Tabular => "tabular",
            Modality::Video => "video",
            Modality::Text => "text",
            _ => continue,
        };
        let file_strs: Vec<String> = files.iter().map(|p| p.to_string_lossy().to_string()).collect();
        paths_map.insert(key.to_string(), serde_json::to_value(file_strs)?);
    }
    Ok(serde_json::Value::Object(paths_map))
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Analyze { paths, report, fusion: _, plot } => {
            println!("═══════════════════════════════════════════════");
            println!("DataSense — Processing Datasets...");
            println!("═══════════════════════════════════════════════");

            let mut ingestor = Ingestor::new();
            for path in paths {
                ingestor.scan(&path).with_context(|| format!("Failed to scan {:?}", path))?;
            }

            for (modality, files) in &ingestor.files {
                println!("Detected {:?}: {} samples", modality, files.len());
            }

            let mut input_map = serde_json::Map::new();
            input_map.insert("action".to_string(), serde_json::Value::String("analyze".to_string()));
            input_map.insert("paths".to_string(), build_paths_input(&ingestor)?);
            
            if plot {
                let plot_dir = std::env::current_dir()?.join("datasense_plots");
                input_map.insert("plot_dir".to_string(), serde_json::Value::String(plot_dir.to_string_lossy().to_string()));
            }

            if let Some(r) = report {
                input_map.insert("report_md".to_string(), serde_json::Value::String(r.to_string_lossy().to_string()));
            }

            let result = run_python(&input_map)?;
            
            println!("\n─── RECOMMENDATION ─────────────────────────────");
            if let Some(recs) = result.get("recommendation") {
                let primary = recs.get("primary").and_then(|v| v.as_str()).unwrap_or("Unknown");
                let confidence = (recs.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.0) * 100.0) as u32;
                println!("PRIMARY    : {} [{}%]", primary, confidence);
                
                if let Some(summary) = result.get("summary").and_then(|v| v.as_str()) {
                    println!("\nAI SUMMARY:\n{}", summary);
                }
            }
            println!("═══════════════════════════════════════════════");
        }
        Commands::Init { path, output } => {
            println!("═══════════════════════════════════════════════");
            println!("DataSense — Building Training Blueprint...");
            println!("═══════════════════════════════════════════════");

            let mut ingestor = Ingestor::new();
            ingestor.scan(&path)?;
            let paths_input = build_paths_input(&ingestor)?;
            
            let mut input_map = serde_json::Map::new();
            input_map.insert("action".to_string(), serde_json::Value::String("init".to_string()));
            input_map.insert("paths".to_string(), paths_input);
            input_map.insert("output_blueprint".to_string(), serde_json::Value::String(output.to_string_lossy().to_string()));
            
            let result = run_python(&input_map)?;
            println!("BLUEPRINT SAVED: {}", output.display());
            if let Some(model) = result.get("recommendation").and_then(|r| r.get("primary")) {
                println!("TARGET MODEL   : {}", model);
            }
            println!("═══════════════════════════════════════════════");
        }
        Commands::Dashboard { path } => {
            let mut ingestor = Ingestor::new();
            ingestor.scan(&path)?;
            let paths_input = build_paths_input(&ingestor)?;
            
            let mut input_map = serde_json::Map::new();
            input_map.insert("action".to_string(), serde_json::Value::String("dashboard".to_string()));
            input_map.insert("paths".to_string(), paths_input);
            
            run_python_raw(&input_map)?; 
        }
        Commands::Compare { path1, path2 } => {
            println!("═══════════════════════════════════════════════");
            println!("DataSense — Comparing Datasets...");
            println!("═══════════════════════════════════════════════");

            let mut ing1 = Ingestor::new();
            ing1.scan(&path1)?;
            let mut ing2 = Ingestor::new();
            ing2.scan(&path2)?;

            println!("MODALITY    | DATASET 1 | DATASET 2");
            println!("------------|-----------|-----------");
            let modalities = [Modality::Image, Modality::Audio, Modality::Video, Modality::Tabular, Modality::Text];
            for m in modalities {
                let count1 = ing1.files.get(&m).map(|v| v.len()).unwrap_or(0);
                let count2 = ing2.files.get(&m).map(|v| v.len()).unwrap_or(0);
                if count1 > 0 || count2 > 0 {
                    println!("{: <11} | {: <9} | {: <9}", format!("{:?}", m), count1, count2);
                }
            }
            println!("═══════════════════════════════════════════════");
        }
        Commands::Score { path, model, plot, report } => {
            let mut ingestor = Ingestor::new();
            ingestor.scan(&path)?;

            let mut input_map = serde_json::Map::new();
            input_map.insert("paths".to_string(), build_paths_input(&ingestor)?);

            match model {
                Some(m_name) => {
                    input_map.insert("action".to_string(), serde_json::Value::String("score".to_string()));
                    input_map.insert("model".to_string(), serde_json::Value::String(m_name));
                    let result = run_python(&input_map)?;
                    if let Some(res) = result.get("score_result") {
                        let s = (res.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0) * 100.0) as u32;
                        println!("MODEL SCORE: {}% - {}", s, res.get("justification").and_then(|v| v.as_str()).unwrap_or(""));
                    }
                }
                None => {
                    input_map.insert("action".to_string(), serde_json::Value::String("score_all".to_string()));
                    if plot {
                        input_map.insert("plot_dir".to_string(), serde_json::Value::String("datasense_plots".to_string()));
                    }
                    if let Some(r) = report {
                        input_map.insert("report_md".to_string(), serde_json::Value::String(r.to_string_lossy().to_string()));
                    }
                    let result = run_python(&input_map)?;
                    if let Some(leaderboard) = result.get("leaderboard").and_then(|v| v.as_array()) {
                        println!("🥇 SCORE | MODEL");
                        for entry in leaderboard {
                            let m = entry.get("model").and_then(|v| v.as_str()).unwrap_or("?");
                            let s = (entry.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0) * 100.0) as u32;
                            println!("{: >5}% | {}", s, m);
                        }
                    }
                }
            }
        }
    }

    Ok(())
}
