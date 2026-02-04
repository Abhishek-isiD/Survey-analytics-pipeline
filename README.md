# Survey-analytics-pipeline
Modular survey analytics pipeline with data cleaning, weighting, ML modeling, PCA outlier detection, and automated reporting.

- Advanced cleaning (MICE, KNN, PCA outliers)
- Survey weighting + raking
- Statistical summaries
- ML modeling (classification + regression)
- Automated reporting

## Features

- Plugin architecture (cleaner / weighting / analysis / reporter)
- Config-driven behavior (JSON)
- Audit logging
- PCA visualization
- ML model comparison
- Prediction + confidence intervals

## Quick Start

pip install -r requirements.txt  
python survey_pipeline.py

## Architecture

SurveyPipeline
 ├── Cleaner
 ├── WeightingEngine
 ├── AnalysisEngine
 └── Reporter
