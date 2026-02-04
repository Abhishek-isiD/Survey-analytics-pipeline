# survey_pipeline.py
import pandas as pd
import numpy as np
import json
from abc import ABC, abstractmethod
import logging
import os
from datetime import datetime

# Set up logging for integration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataPipeline")

# --- Abstract Base Classes (Interfaces) ---
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, source): pass

class DataCleaner(ABC):
    @abstractmethod
    def clean(self, data): pass

class WeightingEngine(ABC):
    @abstractmethod
    def apply_weights(self, data_path): pass

class AnalysisEngine(ABC):
    @abstractmethod
    def analyze(self, data_path: str): pass

class Reporter(ABC):
    @abstractmethod
    def generate_report(self, data_path, estimates, output_path): pass

class SensitivityAnalyzer(ABC):
    @abstractmethod
    def analyze(self, data_path: str, analysis_config: dict) -> dict: pass

# --- Global Constants ---
TEMP_REPORTS_DIR = "reports_temp"
CLEANED_DATA_FILE = os.path.join(TEMP_REPORTS_DIR, "cleaned_data.csv")
WEIGHTED_DATA_FILE = os.path.join(TEMP_REPORTS_DIR, "cleaned_and_weighted_data.csv")
AUDIT_LOG_FILE = os.path.join(TEMP_REPORTS_DIR, "audit_log.txt")


# --- Core Survey Pipeline Class ---
class SurveyPipeline:
    """
    Core pipeline that coordinates between different data processing modules.
    """
    def __init__(self, config_path="pipeline_config.json"):
        self.config = self._load_config(config_path)
        self.ingestor = None
        self.cleaner = None
        self.weighting = None
        self.analysis_engine = None
        self.reporter = None
        self.sensitivity_analyzer = None
        self.raw_data = None
        self._cleaned_data_path = None
        self.weighted_data_path = None
        self.estimates = None
        self.sensitivity_results = None
        self.audit_log = []
        logger.info("Pipeline initialized with configuration")

    def _log_audit(self, message):
        """Helper function to add timestamped messages to the audit log."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.audit_log.append(f"[{timestamp}] - {message}")

    def _load_config(self, path):
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found at '{path}'. Pipeline cannot run without configuration.")
            raise
        except json.JSONDecodeError:
            logger.error(f"Config file at '{path}' is malformed. Please check the JSON syntax.")
            raise

    def register_module(self, module_type, module_instance):
        setattr(self, module_type, module_instance)
        logger.info(f"Registered {module_type} module: {module_instance.__class__.__name__}")

    def ingest_data(self, source):
        self._log_audit(f"START: Data Ingestion from source: {source}")
        logger.info(f"Starting data ingestion from source: {source}")
        try:
            if self.ingestor:
                self.raw_data = self.ingestor.ingest(source)
            elif source.endswith('.csv'):
                self.raw_data = pd.read_csv(source)
            elif source.endswith('.xlsx'):
                self.raw_data = pd.read_excel(source)
            else:
                raise ValueError("Unsupported source format. Please provide .csv or .xlsx.")
            
            self._log_audit(f"SUCCESS: Ingested {len(self.raw_data)} records and {len(self.raw_data.columns)} columns.")
            logger.info(f"Ingested {len(self.raw_data)} records.")
            return True
        except Exception as e:
            self._log_audit(f"FAILURE: Ingestion failed with error: {e}")
            logger.error(f"Ingestion failed: {e}")
            return False

    def clean_data(self):
        if self.raw_data is None:
            logger.error("No raw data to clean. Ingest data first.")
            return None
        
        self._log_audit("START: Data Cleaning")
        cleaner_config = self.config.get("custom_module_settings", {}).get("cleaner", {})
        self._log_audit(f"  - Parameters: {json.dumps(cleaner_config)}")
        self._log_audit(f"  - Initial rows: {len(self.raw_data)}")
        logger.info("Starting data cleaning.")
        try:
            if self.cleaner:
                # The custom cleaner now returns the cleaning summary
                cleaning_summary = self.cleaner.clean(self.raw_data)
                self._cleaned_data_path = CLEANED_DATA_FILE
                
                final_rows = cleaning_summary.get("final_dimensions", {}).get("rows", "N/A")
                self._log_audit(f"  - Final rows: {final_rows}")
                self._log_audit("SUCCESS: Data cleaning completed.")
                logger.info(f"Data cleaned via external module. Artifacts in {TEMP_REPORTS_DIR}")
            else:
                self._cleaned_data_path = self._basic_clean_and_save(self.raw_data)
                self._log_audit("SUCCESS: Basic data cleaning completed.")
                logger.info(f"Applied basic cleaning. Cleaned data at: {self._cleaned_data_path}")
            return self._cleaned_data_path
        except Exception as e:
            self._log_audit(f"FAILURE: Data cleaning failed with error: {e}")
            logger.error(f"Data cleaning failed: {e}", exc_info=True)
            return None

    def _basic_clean_and_save(self, data: pd.DataFrame) -> str:
        cleaned_data = data.dropna().copy()
        os.makedirs(TEMP_REPORTS_DIR, exist_ok=True)
        cleaned_data.to_csv(CLEANED_DATA_FILE, index=False)
        return CLEANED_DATA_FILE

    def apply_weights(self, cleaned_data_path):
        if not cleaned_data_path or not os.path.exists(cleaned_data_path):
            logger.error("No clean data available for weighting.")
            return None
        
        self._log_audit("START: Weighting")
        weighting_config = self.config.get("custom_module_settings", {}).get("weighting", {})
        self._log_audit(f"  - Parameters: {json.dumps(weighting_config)}")
        logger.info(f"Starting weighting on data from: {cleaned_data_path}")
        try:
            if self.weighting:
                _, self.estimates = self.weighting.apply_weights(cleaned_data_path)
                self.weighted_data_path = WEIGHTED_DATA_FILE
                resampling_note = self.estimates.get("weighting_process_log", {}).get("resampling_note", "N/A")
                self._log_audit(f"  - {resampling_note}")
                self._log_audit("SUCCESS: Weighting completed.")
                logger.info(f"Weights applied via external module. Weighted data saved to: {self.weighted_data_path}")
                return self.weighted_data_path
            else:
                # Basic fallback weighting
                data = pd.read_csv(cleaned_data_path)
                weight_col = self.config.get('weight_column', 'weight')
                if weight_col not in data.columns:
                    data[weight_col] = 1
                data.to_csv(WEIGHTED_DATA_FILE, index=False)
                self.weighted_data_path = WEIGHTED_DATA_FILE
                self._log_audit("SUCCESS: Basic weighting completed.")
                logger.info(f"Applied basic weighting. Weighted data at: {self.weighted_data_path}")
                return self.weighted_data_path
        except Exception as e:
            self._log_audit(f"FAILURE: Weighting failed with error: {e}")
            logger.error(f"Weighting module failed: {e}", exc_info=True)
            return None

    def run_analysis(self, weighted_data_path: str):
        if not self.analysis_engine or not weighted_data_path or not os.path.exists(weighted_data_path):
            logger.info("Skipping modeling analysis (module not registered or no weighted data).")
            return False
        
        self._log_audit("START: Modeling Analysis")
        analysis_config = self.config.get("custom_module_settings", {}).get("analysis_config", {})
        self._log_audit(f"  - Parameters: {json.dumps(analysis_config)}")
        logger.info("Starting modeling analysis.")
        try:
            self.analysis_engine.analyze(weighted_data_path)
            self._log_audit("SUCCESS: Modeling analysis completed.")
            logger.info("Modeling analysis completed successfully.")
            return True
        except Exception as e:
            self._log_audit(f"FAILURE: Modeling analysis failed with error: {e}")
            logger.error(f"Modeling analysis failed: {e}", exc_info=True)
            return False

    def generate_report(self, output_path="survey_output"):
        self._log_audit("START: Report Generation")
        if not self.reporter:
            self._log_audit("FAILURE: No reporter module registered.")
            logger.error("No reporter module available for reporting.")
            return None
        logger.info("Starting report generation.")
        try:
            # First, write the audit log to the temp directory
            with open(AUDIT_LOG_FILE, 'w') as f:
                f.write("\n".join(self.audit_log))

            # The reporter will now pick up the audit log along with other artifacts
            report_final_path = self.reporter.generate_report(
                data_path=self._cleaned_data_path,
                estimates=self.estimates,
                output_path=output_path
            )
            self._log_audit(f"SUCCESS: Report generated at {report_final_path}")
            logger.info(f"Report generated via external module: {report_final_path}")
            return report_final_path
        except Exception as e:
            self._log_audit(f"FAILURE: Report generation failed with error: {e}")
            logger.error(f"External reporter failed: {e}", exc_info=True)
            return None

    def run_pipeline(self, source, output_path="survey_output"):
        self._log_audit(f"--- START: Full Pipeline Run for source: {source} ---")
        results = {
            'ingestion_success': False, 
            'cleaning_success': False, 
            'weighting_success': False, 
            'analysis_success': False, 
            'reporting_path': None
        }
        if self.ingest_data(source):
            results['ingestion_success'] = True
            cleaned_data_path = self.clean_data()
            if cleaned_data_path:
                results['cleaning_success'] = True
                weighted_data_path = self.apply_weights(cleaned_data_path)
                if weighted_data_path:
                    results['weighting_success'] = True
                    if self.run_analysis(weighted_data_path):
                         results['analysis_success'] = True
        
        report_path = self.generate_report(output_path)
        if report_path: results['reporting_path'] = report_path
        
        self._log_audit(f"--- END: Pipeline completed with final status: {results} ---")
        logger.info(f"--- Pipeline completed with final status: {results} ---")
        return results

# --- Example Usage ---
if __name__ == "__main__":
    try:
        pipeline = SurveyPipeline("pipeline_config.json")
        from custom_modules import (MyCustomCleaner, MyCustomWeightingEngine, 
                                    MyCustomFolderReporter, MyCustomSensitivityAnalyzer, 
                                    MyCustomAnalysisEngine)
        
        pipeline.register_module('cleaner', MyCustomCleaner())
        pipeline.register_module('weighting', MyCustomWeightingEngine())
        pipeline.register_module('analysis_engine', MyCustomAnalysisEngine())
        pipeline.register_module('reporter', MyCustomFolderReporter())
        pipeline.register_module('sensitivity_analyzer', MyCustomSensitivityAnalyzer())
        logger.info("All custom modules successfully imported and registered.")

        final_pipeline_results = pipeline.run_pipeline("survey_data.csv", "my_final_survey_report")

        print("\n--- Pipeline Run Summary ---")
        for stage, status in final_pipeline_results.items():
            print(f"{stage.replace('_', ' ').capitalize()}: {status}")

        if final_pipeline_results.get('reporting_path'):
            print(f"\nReport folder generated at: {final_pipeline_results['reporting_path']}")
        else:
            print("\nNo report folder was generated due to earlier pipeline failures.")
            
    except Exception as e:
        logger.critical(f"A critical error occurred during pipeline setup or execution: {e}", exc_info=True)
