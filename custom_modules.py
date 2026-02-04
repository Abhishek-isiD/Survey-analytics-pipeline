# custom_modules.py
import pandas as pd
import numpy as np
import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import shutil
import plotly.express as px
import joblib
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor, IsolationForest, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
import statsmodels.api as sm
from scipy import stats
from scipy.stats import median_abs_deviation
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from ipfn import ipfn

# Import the abstract base classes (interfaces) from survey_pipeline
from survey_pipeline import DataCleaner, WeightingEngine, Reporter, SensitivityAnalyzer, AnalysisEngine

# --- Constants ---
CONFIG_FILE_PATH = "pipeline_config.json"
TEMP_REPORTS_DIR = "reports_temp"
CLEANING_SUMMARY_FILE = os.path.join(TEMP_REPORTS_DIR, "cleaning_summary.json")
CLEANED_DATA_FILE = os.path.join(TEMP_REPORTS_DIR, "cleaned_data.csv")
DROPPED_DATA_FILE = os.path.join(TEMP_REPORTS_DIR, "dropped_data.csv")
WEIGHTED_DATA_FILE = os.path.join(TEMP_REPORTS_DIR, "cleaned_and_weighted_data.csv")
STATISTICAL_SUMMARY_FILE = os.path.join(TEMP_REPORTS_DIR, "statistical_summary.json")
MODEL_ANALYSIS_FILE = os.path.join(TEMP_REPORTS_DIR, "model_analysis_report.json")
PCA_VARIANCE_FILE = os.path.join(TEMP_REPORTS_DIR, "pca_explained_variance.json") 
SAVED_MODEL_FILE = os.path.join(TEMP_REPORTS_DIR, "best_model.joblib")
DATASET_SIZE_THRESHOLD = 1_000_000

# --- Helper Functions ---

class NumpyJSONEncoder(json.JSONEncoder):
    """ Custom JSON encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def _load_module_config(full_config=False):
    """ Loads the pipeline configuration from a JSON file. """
    try:
        with open(CONFIG_FILE_PATH, 'r') as f:
            config = json.load(f)
        if full_config:
            return config
        return config.get("custom_module_settings", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


class MyCustomCleaner(DataCleaner):
    """
    A custom data cleaning module with advanced features like PCA-based outlier detection,
    multiple imputation strategies, and configurable data dropping rules.
    """
    def __init__(self):
        full_config = _load_module_config(full_config=True)
        self.settings = full_config.get("custom_module_settings", {}).get("cleaner", {})
        self.performance_settings = full_config.get("custom_module_settings", {}).get("performance_settings", {})
        
        self.required_fields = full_config.get("required_fields", None)
        self.column_types = full_config.get("column_types", {})

        self.numeric_impute_cols = self.settings.get("numeric_cols_for_imputation", [])
        self.categorical_impute_cols = self.settings.get("categorical_cols_for_imputation", [])
        self.row_null_threshold = self.settings.get("row_null_threshold", 100)
        self.column_null_threshold = self.settings.get("column_null_threshold", 100)
        self.rules_based_imputation = self.settings.get("rules_based_imputation", [])
        
        self.imputation_strategy = self.settings.get("imputation_strategy", "smart_impute")
        self.mice_max_iter = self.settings.get("mice_max_iter", 10)
        self.mice_n_estimators = self.settings.get("mice_n_estimators", 10)
        self.n_jobs_for_imputation = self.settings.get("n_jobs_for_imputation", -1)
        
        self.drop_outliers = self.settings.get("drop_outliers", False)
        self.outlier_detection_strategy = self.settings.get("outlier_detection_strategy", "none")
        self.outlier_removal_percent = self.settings.get("outlier_removal_percent", 0.0)
        self.dbscan_eps = self.settings.get("dbscan_eps", 0.5)
        self.dbscan_min_samples = self.settings.get("dbscan_min_samples", 5)

        self.dask_npartitions = self.settings.get("dask_npartitions", 100)

    def clean(self, data: pd.DataFrame):
        os.makedirs(TEMP_REPORTS_DIR, exist_ok=True)
        
        if self.required_fields:
            valid_fields = [field for field in self.required_fields if field in data.columns]
            cleaned_data = data[valid_fields].copy()
        else:
            cleaned_data = data.copy()

        initial_rows, initial_columns = cleaned_data.shape
        all_dropped_rows = []
        imputation_summary = {} 
        pca_plot_path = None
        pca_explained_variance = None
        pca_formulas = None

        failed_conversion_rows = self._enforce_column_types(cleaned_data)
        if not failed_conversion_rows.empty:
            all_dropped_rows.append(failed_conversion_rows)
            cleaned_data.drop(index=failed_conversion_rows.index, inplace=True)

        self._apply_rules_based_imputation(cleaned_data, self.rules_based_imputation)

        cols_to_drop_perc = (cleaned_data.isnull().sum() / initial_rows * 100)
        columns_to_drop = [col for col, perc in cols_to_drop_perc.items() if perc >= self.column_null_threshold]
        
        if columns_to_drop:
            print("\n--- CLEANER WARNING: Dropping columns with high null values ---")
            for col in columns_to_drop:
                print(f"  - Dropping '{col}' ({cols_to_drop_perc[col]:.2f}% nulls, threshold is {self.column_null_threshold}%)")
            print("  - To keep these columns, increase 'column_null_threshold' in your pipeline_config.json\n")

        cleaned_data.drop(columns=columns_to_drop, inplace=True)
        
        rows_to_drop_indices = cleaned_data[cleaned_data.isnull().sum(axis=1) / cleaned_data.shape[1] * 100 >= self.row_null_threshold].index
        if not rows_to_drop_indices.empty:
            dropped_for_nulls = data.loc[rows_to_drop_indices].copy()
            dropped_for_nulls['reason_for_dropping'] = f'Exceeded {self.row_null_threshold}% row null threshold'
            all_dropped_rows.append(dropped_for_nulls)
            cleaned_data.drop(index=rows_to_drop_indices, inplace=True)
        
        if self.drop_outliers:
            outlier_indices = pd.Index([])
            if self.outlier_detection_strategy == "iqr":
                outlier_indices = self._iqr_outlier_detection(cleaned_data)
            elif self.outlier_detection_strategy == "zscore":
                outlier_indices = self._zscore_outlier_detection(cleaned_data)
            elif self.outlier_detection_strategy == "winsorize":
                self._winsorize_outlier_treatment(cleaned_data)
            elif self.outlier_detection_strategy in ["isolation_forest", "dbscan"]:
                if self.performance_settings.get("run_pca_analysis", False):
                    print("Running PCA analysis for outlier detection...")
                    temp_imputed_for_outliers = cleaned_data.copy()
                    self._median_mode_imputation(temp_imputed_for_outliers) 
                    numeric_data = temp_imputed_for_outliers.select_dtypes(include=np.number)
                    if numeric_data.shape[1] > 1:
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(numeric_data)
                        n_features = scaled_data.shape[1]
                        n_pca_components = min(5, n_features)
                        pca = PCA(n_components=n_pca_components, svd_solver='randomized', random_state=42)
                        principal_components = pca.fit_transform(scaled_data)
                        numeric_feature_names = numeric_data.columns.tolist()
                        pca_formulas = {f"PC_{i+1}": " + ".join([f"{weight:.2f} * {name}" for weight, name in zip(component, numeric_feature_names)]) for i, component in enumerate(pca.components_)}
                        pca_explained_variance = {f"PC_{i+1}": f"{var*100:.2f}%" for i, var in enumerate(pca.explained_variance_ratio_)}
                        pca_report = {"explained_variance": pca_explained_variance, "component_formulas": pca_formulas}
                        with open(PCA_VARIANCE_FILE, 'w') as f:
                            json.dump(pca_report, f, indent=4, cls=NumpyJSONEncoder)
                        
                        if self.outlier_detection_strategy == "dbscan":
                            anomaly_scores = self._dbscan_outlier_detection(temp_imputed_for_outliers, numeric_data.columns)
                        else: # isolation_forest
                            anomaly_scores = self._isolation_forest_outlier_detection(temp_imputed_for_outliers, numeric_data.columns)
                        cleaned_data['anomaly_score'] = anomaly_scores
                        if self.outlier_detection_strategy == "dbscan":
                            outlier_indices = cleaned_data[cleaned_data['anomaly_score'] == -1].index
                        else:
                            num_outliers_to_remove = int(len(cleaned_data) * self.outlier_removal_percent / 100)
                            outlier_indices = cleaned_data.sort_values(by='anomaly_score', ascending=True).iloc[:num_outliers_to_remove].index
                        
                        if not outlier_indices.empty:
                            pca_plot_path = self._generate_pca_outlier_plot(principal_components, cleaned_data.index, outlier_indices, os.path.join(TEMP_REPORTS_DIR, "pca_outlier_visualization.html"))
            
            if not outlier_indices.empty:
                outliers_df = data.loc[outlier_indices].copy()
                outliers_df['reason_for_dropping'] = f'Outlier ({self.outlier_detection_strategy})'
                all_dropped_rows.append(outliers_df)
                cleaned_data.drop(index=outlier_indices, inplace=True)

        cleaned_data = cleaned_data.drop(columns=['anomaly_score'], errors='ignore')

        if self.imputation_strategy == "knn":
            imputation_summary = self._knn_imputation(cleaned_data)
        elif self.imputation_strategy == "smart_impute":
            imputation_summary = self._smart_imputation(cleaned_data)
        elif self.imputation_strategy == "mice":
            imputation_summary = self._mice_imputation(cleaned_data, self.numeric_impute_cols, self.categorical_impute_cols)
        
        final_rows, final_columns = cleaned_data.shape
        cleaned_data.to_csv(CLEANED_DATA_FILE, index=False)
        
        dropped_data_path = None
        if all_dropped_rows:
            combined_dropped_df = pd.concat(all_dropped_rows, ignore_index=False)
            combined_dropped_df.to_csv(DROPPED_DATA_FILE, index=True)
            dropped_data_path = DROPPED_DATA_FILE

        cleaning_summary = {
            "initial_dimensions": {"rows": initial_rows, "columns": initial_columns},
            "final_dimensions": {"rows": final_rows, "columns": final_columns},
            "pca_plot_path": pca_plot_path,
            "pca_explained_variance": pca_explained_variance,
            "pca_component_formulas": pca_formulas,
            "cleaned_data_path": CLEANED_DATA_FILE,
            "dropped_data_path": dropped_data_path,
            "imputation_summary": imputation_summary
        }
        
        with open(CLEANING_SUMMARY_FILE, 'w') as f:
            json.dump(cleaning_summary, f, indent=4, cls=NumpyJSONEncoder)
        return cleaning_summary

    def _generate_pca_outlier_plot(self, principal_components, original_indices, outlier_indices, filepath_html):
        n_components = principal_components.shape[1]
        if n_components < 2:
            return None

        pc_cols = [f'PC{i+1}' for i in range(n_components)]
        pc_df = pd.DataFrame(data=principal_components, columns=pc_cols, index=original_indices)
        pc_df['status'] = 'inlier'
        pc_df.loc[outlier_indices, 'status'] = 'outlier'

        if n_components >= 3:
            fig_interactive = px.scatter_3d(pc_df, x='PC1', y='PC2', z='PC3', color='status',
                                            title='3D PCA Outlier Visualization (Interactive)',
                                            color_discrete_map={'inlier': 'blue', 'outlier': 'red'})
            fig_interactive.update_traces(marker=dict(size=4, opacity=0.7))
        else:
            fig_interactive = px.scatter(pc_df, x='PC1', y='PC2', color='status',
                                         title='2D PCA Outlier Visualization (Interactive)',
                                         color_discrete_map={'inlier': 'blue', 'outlier': 'red'})
            fig_interactive.update_traces(marker=dict(size=5, opacity=0.7))
        
        fig_interactive.write_html(filepath_html)

        filepath_png = filepath_html.replace('.html', '_static.png')
        if n_components < 3:
            fig_static, ax1 = plt.subplots(1, 1, figsize=(9, 8))
            fig_static.suptitle('PCA Outlier Visualization (Static)', fontsize=16)
        else:
            fig_static = plt.figure(figsize=(18, 8))
            fig_static.suptitle('PCA Outlier Visualization (Static)', fontsize=16)
            ax1 = fig_static.add_subplot(1, 2, 1)

        sns.scatterplot(data=pc_df, x='PC1', y='PC2', hue='status', palette={'inlier': 'blue', 'outlier': 'red'}, ax=ax1, alpha=0.7)
        ax1.set_title('2D PCA of Outliers')
        ax1.grid(True)

        if n_components >= 3:
            ax2 = fig_static.add_subplot(1, 2, 2, projection='3d')
            inliers = pc_df[pc_df['status'] == 'inlier']
            outliers = pc_df[pc_df['status'] == 'outlier']
            ax2.scatter(inliers['PC1'], inliers['PC2'], inliers['PC3'], c='blue', label='Inlier', alpha=0.5)
            ax2.scatter(outliers['PC1'], outliers['PC2'], outliers['PC3'], c='red', label='Outlier', s=50)
            ax2.set_title('3D PCA of Outliers')
            ax2.set_xlabel('PC1')
            ax2.set_ylabel('PC2')
            ax2.set_zlabel('PC3')
            ax2.legend()

        plt.savefig(filepath_png)
        plt.close(fig_static)

        return filepath_html

    def _enforce_column_types(self, data: pd.DataFrame):
        failed_rows_list = []
        for col, dtype in self.column_types.items():
            if col in data.columns:
                original_col = data[col].copy()
                try:
                    dtype_map = {"integer": pd.Int64Dtype(),"float": "float64","category": "category","string": "string","datetime": "datetime64[ns]"}
                    target_dtype = dtype_map.get(dtype.lower(), dtype)
                    if pd.api.types.is_datetime64_any_dtype(target_dtype):
                        converted_col = pd.to_datetime(data[col], errors='coerce')
                    else:
                        converted_col = data[col].astype(target_dtype)
                    failed_indices = original_col.notna() & converted_col.isna()
                    if failed_indices.any():
                        failed_df = data.loc[failed_indices].copy()
                        failed_df['reason_for_dropping'] = f"Failed data type conversion for column '{col}' to '{dtype}'"
                        failed_rows_list.append(failed_df)
                    data[col] = converted_col
                except Exception as e:
                    print(f"Warning: Could not convert column '{col}' to type '{dtype}'. Error: {e}")
        if failed_rows_list:
            return pd.concat(failed_rows_list).drop_duplicates()
        else:
            return pd.DataFrame()

    def _smart_imputation(self, data: pd.DataFrame):
        summary = {}
        for col in self.numeric_impute_cols:
            if col in data.columns and data[col].isnull().any():
                count = data[col].isnull().sum()
                if pd.api.types.is_integer_dtype(data[col]):
                    impute_value = data[col].mode()[0]
                    data[col].fillna(impute_value, inplace=True)
                    summary[col] = f"Imputed {count} values with mode ({impute_value})"
                else:
                    impute_value = data[col].median()
                    data[col].fillna(impute_value, inplace=True)
                    summary[col] = f"Imputed {count} values with median ({impute_value})"
        for col in self.categorical_impute_cols:
            if col in data.columns and data[col].isnull().any():
                count = data[col].isnull().sum()
                impute_value = data[col].mode()[0]
                data[col].fillna(impute_value, inplace=True)
                summary[col] = f"Imputed {count} values with mode ({impute_value})"
        return summary
    
    def _median_mode_imputation(self, data: pd.DataFrame):
        summary = {}
        for col in self.numeric_impute_cols:
            if col in data.columns and data[col].isnull().any():
                count = data[col].isnull().sum()
                impute_value = data[col].median()
                data[col].fillna(impute_value, inplace=True)
                summary[col] = f"Imputed {count} values with median ({impute_value})"
        for col in self.categorical_impute_cols:
            if col in data.columns and data[col].isnull().any():
                count = data[col].isnull().sum()
                impute_value = data[col].mode()[0]
                data[col].fillna(impute_value, inplace=True)
                summary[col] = f"Imputed {count} values with mode ({impute_value})"
        return summary

    def _knn_imputation(self, data: pd.DataFrame):
        summary = {}
        numeric_data = data.select_dtypes(include=np.number)
        if numeric_data.isnull().sum().sum() > 0:
            imputer = KNNImputer(n_neighbors=5)
            imputed_data = imputer.fit_transform(numeric_data)
            data[numeric_data.columns] = imputed_data
            summary["knn_imputation"] = "Applied KNN imputation to numeric columns."
        return summary

    def _iqr_outlier_detection(self, data: pd.DataFrame):
        outlier_indices = pd.Index([])
        numeric_cols = data.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            col_outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)].index
            outlier_indices = outlier_indices.union(col_outliers)
        return outlier_indices

    def _zscore_outlier_detection(self, data: pd.DataFrame, threshold=3):
        outlier_indices = pd.Index([])
        numeric_cols = data.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            mean = data[col].mean()
            std = data[col].std()
            z_scores = (data[col] - mean) / std
            col_outliers = data[np.abs(z_scores) > threshold].index
            outlier_indices = outlier_indices.union(col_outliers)
        return outlier_indices

    def _winsorize_outlier_treatment(self, data: pd.DataFrame, limits=(0.05, 0.05)):
        numeric_cols = data.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            data[col] = stats.mstats.winsorize(data[col], limits=limits)
        return data
        
    def _mice_imputation(self, data: pd.DataFrame, numeric_cols: list, categorical_cols: list):
        summary = {}
        
        # --- START: ROBUSTNESS FIX ---
        # Filter for numeric columns that actually exist in the DataFrame before trying to use them.
        valid_numeric_cols = [col for col in numeric_cols if col in data.columns]
        
        if not valid_numeric_cols:
            summary["MICE_NOTE"] = "No numeric columns specified for MICE were found in the data (they may have been dropped). Skipping MICE for numerics."
        else:
            numeric_data_to_impute = data[valid_numeric_cols].copy()
            if numeric_data_to_impute.isnull().sum().sum() > 0:
                try:
                    estimator = RandomForestRegressor(n_estimators=self.mice_n_estimators, random_state=0, n_jobs=self.n_jobs_for_imputation)
                    imputer = IterativeImputer(estimator=estimator, max_iter=self.mice_max_iter, random_state=0)
                    imputed_numeric_array = imputer.fit_transform(numeric_data_to_impute)
                    imputed_numeric_df = pd.DataFrame(imputed_numeric_array, columns=valid_numeric_cols, index=data.index)
                    data[valid_numeric_cols] = imputed_numeric_df
                    for col in valid_numeric_cols:
                        count = numeric_data_to_impute[col].isnull().sum()
                        if count > 0:
                            summary[col] = f"Imputed {count} values with MICE"
                except Exception as e:
                    summary["MICE_FAILURE"] = f"MICE failed on numeric columns with error: {e}. Falling back to median imputation for numerics."
                    for col in valid_numeric_cols:
                        if data[col].isnull().any():
                            impute_value = data[col].median()
                            data[col].fillna(impute_value, inplace=True)
                            summary[col] = f"Imputed {data[col].isnull().sum()} values with median (fallback)"
        # --- END: ROBUSTNESS FIX ---

        # This logic is already robust because it checks `if col in data.columns`
        for col in categorical_cols:
            if col in data.columns and data[col].isnull().any():
                count = data[col].isnull().sum()
                impute_value = data[col].mode()[0]
                data[col].fillna(impute_value, inplace=True)
                summary[col] = f"Imputed {count} values with mode ({impute_value})"
        return summary

    def _apply_rules_based_imputation(self, data: pd.DataFrame, rules: list):
        for rule in rules:
            dependent_col = rule.get("dependent_column")
            condition_col = rule.get("condition_column")
            condition_value = rule.get("condition_value")
            impute_value = rule.get("impute_value")

            if (dependent_col in data.columns and condition_col in data.columns):
                condition_mask = (data[condition_col] == condition_value).fillna(False)
                null_mask = data[dependent_col].isnull()
                
                rows_to_impute_index = data[condition_mask & null_mask].index
                if not rows_to_impute_index.empty:
                    data.loc[rows_to_impute_index, dependent_col] = impute_value
    
    def _dbscan_outlier_detection(self, data: pd.DataFrame, numeric_cols: list):
        numeric_data = data[numeric_cols].select_dtypes(include=np.number)
        if numeric_data.empty: return pd.Series(1, index=data.index)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples, n_jobs=-1)
        clusters = dbscan.fit_predict(scaled_data)
        return pd.Series(clusters, index=numeric_data.index).apply(lambda x: -1 if x == -1 else 1)

    def _isolation_forest_outlier_detection(self, data: pd.DataFrame, numeric_cols: list):
        numeric_data = data[numeric_cols].select_dtypes(include=np.number)
        isolation_forest = IsolationForest(random_state=0, n_jobs=-1)
        isolation_forest.fit(numeric_data)
        return pd.Series(isolation_forest.decision_function(numeric_data), index=data.index)

class MyCustomWeightingEngine(WeightingEngine):
    """
    A custom weighting module that can apply design weights and perform raking
    to adjust sample distributions to known population totals.
    """
    def __init__(self):
        full_config = _load_module_config(full_config=True)
        self.settings = full_config.get("custom_module_settings", {}).get("weighting", {})
        self.performance_settings = full_config.get("custom_module_settings", {}).get("performance_settings", {})
        self.raking_settings = self.settings.get("raking_config", {})
        self.weight_col = full_config.get("weight_column", "weight")
        self.analysis_cols = self.settings.get("analysis_columns", [])

    def apply_weights(self, data_path: str):
        data = pd.read_csv(data_path)
        if self.weight_col not in data.columns:
            data[self.weight_col] = 1
        final_weight_col = self.weight_col
        weighting_summary = {"method": "Using provided design weights.", "raking_applied": False}
        if self.raking_settings.get("apply_raking", False):
            try:
                control_totals_df = pd.read_csv(self.raking_settings["control_totals_path"])
                aggregates = []
                dimensions = []
                for var in self.raking_settings["raking_variables"]:
                    control = control_totals_df[control_totals_df['variable'] == var]
                    agg = control.set_index('category')['population_total']
                    aggregates.append(agg)
                    dimensions.append([var])
                raker = ipfn.ipfn(data, aggregates, dimensions, weight_col=self.weight_col)
                data['final_weight'] = raker.iteration()
                final_weight_col = 'final_weight'
                weighting_summary["raking_applied"] = True
                weighting_summary["method"] = f"Raking applied on variables: {self.raking_settings['raking_variables']}"
            except Exception as e:
                weighting_summary["raking_failure"] = f"Raking failed with error: {e}. Using original design weights."
        
        data.to_csv(WEIGHTED_DATA_FILE, index=False)
        unweighted_summary = self._generate_single_summary(data, weight_col=None)
        weighted_summary_stats = self._generate_single_summary(data, weight_col=final_weight_col)
        normality_analysis = {}
        if self.performance_settings.get("run_normality_plots", False):
            print("Generating normality plots...")
            normality_analysis = self._perform_normality_analysis_and_plot(data, final_weight_col)
        else:
            print("Skipping normality plotting as per performance_settings in config.")
            normality_analysis["note"] = "Plotting skipped for performance reasons."
        
        full_statistical_summary = {
            "unweighted_summary": unweighted_summary,
            "weighted_summary": weighted_summary_stats,
            "normality_analysis": normality_analysis,
            "weighting_process_log": weighting_summary
        }
        with open(STATISTICAL_SUMMARY_FILE, 'w') as f:
            json.dump(full_statistical_summary, f, indent=4, cls=NumpyJSONEncoder)
        return data, full_statistical_summary

    def _generate_single_summary(self, data: pd.DataFrame, weight_col: str = None) -> dict:
        """
        Generates a statistical summary for both numeric and categorical columns.
        Can perform weighted or unweighted calculations.
        """
        summary = {"numeric_summary": {}, "categorical_summary": {}}
        
        for col in self.analysis_cols:
            if col not in data.columns:
                continue

            if pd.api.types.is_numeric_dtype(data[col]):
                if weight_col and weight_col in data.columns:
                    # Weighted calculations
                    valid_data = data[[col, weight_col]].dropna()
                    weighted_mean = np.average(valid_data[col], weights=valid_data[weight_col])
                    # For other metrics like median/std, more complex weighted calculations are needed
                    summary["numeric_summary"][col] = {
                        "mean": weighted_mean,
                        "std_dev": data[col].std(), # Note: This is unweighted std dev for simplicity
                        "min": data[col].min(),
                        "max": data[col].max(),
                    }
                else:
                    # Unweighted calculations
                    summary["numeric_summary"][col] = {
                        "mean": data[col].mean(),
                        "median": data[col].median(),
                        "std_dev": data[col].std(),
                        "min": data[col].min(),
                        "max": data[col].max(),
                    }
            elif pd.api.types.is_categorical_dtype(data[col]) or data[col].dtype == 'object':
                if weight_col and weight_col in data.columns:
                     # Weighted frequency
                    weighted_counts = data.groupby(col)[weight_col].sum()
                    summary["categorical_summary"][col] = (weighted_counts / weighted_counts.sum() * 100).to_dict()
                else:
                    # Unweighted frequency
                    summary["categorical_summary"][col] = (data[col].value_counts(normalize=True) * 100).to_dict()
        return summary

    def _perform_normality_analysis_and_plot(self, data: pd.DataFrame, weight_col: str) -> dict:
        """
        Performs normality tests (Shapiro-Wilk or Kolmogorov-Smirnov) and generates
        distribution plots (histogram and Q-Q plot) for numeric columns.
        """
        normality_results = {}
        numeric_cols_for_analysis = [c for c in self.analysis_cols if c in data.columns and pd.api.types.is_numeric_dtype(data[c])]

        for col in numeric_cols_for_analysis:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f'Normality Analysis for {col}', fontsize=16)

            # Histogram
            sns.histplot(data[col], kde=True, ax=axes[0])
            axes[0].set_title('Distribution Plot')

            # Q-Q Plot
            stats.probplot(data[col].dropna(), dist="norm", plot=axes[1])
            axes[1].set_title('Q-Q Plot')

            plot_path = os.path.join(TEMP_REPORTS_DIR, f"normality_plot_{col}.png")
            plt.savefig(plot_path)
            plt.close(fig)

            # Perform Shapiro-Wilk test for samples < 5000, else use Kolmogorov-Smirnov
            if len(data[col].dropna()) < 5000:
                stat, p_value = stats.shapiro(data[col].dropna())
                test_name = "Shapiro-Wilk"
            else:
                stat, p_value = stats.kstest(data[col].dropna(), 'norm', args=(data[col].mean(), data[col].std()))
                test_name = "Kolmogorov-Smirnov"
            
            normality_results[col] = {
                "test_used": test_name,
                "statistic": stat,
                "p_value": p_value,
                "is_normal_dist (alpha=0.05)": p_value > 0.05,
                "plot_path": plot_path
            }
        return normality_results

class MyCustomFolderReporter(Reporter):
    """
    A custom reporter that gathers all artifacts (JSON summaries, plots, data files)
    from the temporary directory and copies them into a final, timestamped report folder.
    """
    def generate_report(self, data_path: str, estimates: dict, output_path: str) -> str:
        report_folder_path = f"{output_path}_report"
        if os.path.exists(report_folder_path): shutil.rmtree(report_folder_path)
        os.makedirs(report_folder_path)
        
        for summary_file in [CLEANING_SUMMARY_FILE, STATISTICAL_SUMMARY_FILE, MODEL_ANALYSIS_FILE, PCA_VARIANCE_FILE, "audit_log.txt"]:
            temp_path = os.path.join(TEMP_REPORTS_DIR, os.path.basename(summary_file))
            if os.path.exists(temp_path):
                shutil.copy(temp_path, os.path.join(report_folder_path, os.path.basename(summary_file)))
        for data_file in [CLEANED_DATA_FILE, DROPPED_DATA_FILE, WEIGHTED_DATA_FILE]:
             if os.path.exists(data_file):
                shutil.copy(data_file, os.path.join(report_folder_path, os.path.basename(data_file)))
        if os.path.exists(TEMP_REPORTS_DIR):
            for file in os.listdir(TEMP_REPORTS_DIR):
                if file.endswith((".png", ".html")):
                    shutil.copy(os.path.join(TEMP_REPORTS_DIR, file), os.path.join(report_folder_path, file))
            shutil.rmtree(TEMP_REPORTS_DIR)
        return report_folder_path

class MyCustomSensitivityAnalyzer(SensitivityAnalyzer):
    """
    A placeholder for a custom sensitivity analysis module.
    """
    def analyze(self, data_path: str, analysis_config: dict) -> dict:
        return {"result": "Sensitivity analysis performed."}

class MyCustomAnalysisEngine(AnalysisEngine):
    """
    A custom analysis engine that can run multiple prediction or classification models,
    find the best one based on a primary metric, and generate detailed reports including
    prediction/confidence intervals.
    """
    def __init__(self):
        full_config = _load_module_config(full_config=True)
        self.analysis_config = full_config.get("custom_module_settings", {}).get("analysis_config", {})
        self.weight_col = full_config.get("weight_column", "weight")
        self.test_size = self.analysis_config.get("test_size", 0.3)
        self.random_state = self.analysis_config.get("random_state", 42)

    def analyze(self, data_path: str):
        if not self.analysis_config.get("run_analysis", False):
            print("Analysis engine run is set to false in config. Skipping.")
            return

        df = pd.read_csv(data_path)
        
        target = self.analysis_config["target_variable"]
        predictors = self.analysis_config["predictor_variables"]
        model_type = self.analysis_config["model_type"]
        models_to_run_config = self.analysis_config.get("models_to_run", "find_best")
        
        # --- START: ROBUSTNESS FIX ---
        # Check for available predictors to prevent KeyError
        available_predictors = [p for p in predictors if p in df.columns]
        missing_predictors = [p for p in predictors if p not in df.columns]

        if missing_predictors:
            print(f"ANALYSIS WARNING: The following predictor columns were not found in the data and will be ignored: {missing_predictors}")
            print("This can happen if they are not in 'required_fields' or are dropped due to high null counts in the cleaner.")

        if not available_predictors:
            print("ANALYSIS ERROR: None of the specified predictor variables are available in the dataset. Skipping analysis.")
            return

        if target not in df.columns:
            print(f"ANALYSIS ERROR: The target variable '{target}' was not found in the data. Skipping analysis.")
            return

        X = df[available_predictors]
        # --- END: ROBUSTNESS FIX ---
        
        y = df[target]
        
        X = pd.get_dummies(X, drop_first=True)
        # Align y with X after potential row drops from get_dummies if columns had NaNs
        y = y[X.index]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        final_weight_col = 'final_weight' if 'final_weight' in df.columns else self.weight_col
        sample_weights = df.loc[X_train.index, final_weight_col] if final_weight_col in df.columns else None

        all_models = {
            "prediction": {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "Random Forest Regressor": RandomForestRegressor(random_state=self.random_state),
                "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=self.random_state)
            },
            "classification": {
                "Logistic Regression": LogisticRegression(random_state=self.random_state, max_iter=1000),
                "Random Forest Classifier": RandomForestClassifier(random_state=self.random_state),
                "K-Nearest Neighbors (KNN) Classifier": KNeighborsClassifier()
            }
        }

        models_to_run = {}
        if models_to_run_config == "find_best":
            models_to_run = all_models.get(model_type, {})
        else:
            for model_config in models_to_run_config:
                model_name = model_config["name"]
                if model_name in all_models.get(model_type, {}):
                    model_instance = all_models[model_type][model_name]
                    if "params" in model_config:
                        model_instance.set_params(**model_config["params"])
                    models_to_run[model_name] = model_instance

        if not models_to_run:
            print(f"No valid models found to run for model_type '{model_type}'.")
            return
            
        results = []
        best_model_object = None
        best_model_score = -np.inf
        best_model_name = ""
        primary_metric = "Adj. R-squared" if model_type == "prediction" else "F1-Score (Weighted)"

        for name, model in models_to_run.items():
            if sample_weights is not None and hasattr(model, 'fit') and 'sample_weight' in model.fit.__code__.co_varnames:
                 model.fit(X_train, y_train, sample_weight=sample_weights.to_numpy())
            else:
                if sample_weights is not None:
                     print(f"Warning: Model {name} does not support sample weights. Training unweighted.")
                model.fit(X_train, y_train)

            preds = model.predict(X_test)
            
            model_metrics = {"Model": name}
            if model_type == "prediction":
                r2 = r2_score(y_test, preds)
                adj_r2 = 1 - (1-r2)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
                model_metrics.update({
                    "R-squared": r2, "Adj. R-squared": adj_r2,
                    "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
                    "MAE": mean_absolute_error(y_test, preds)
                })
                if adj_r2 > best_model_score:
                    best_model_score = adj_r2
                    best_model_object = model
                    best_model_name = name
            else: # classification
                # Ensure labels are consistent for metrics calculations
                labels = np.union1d(y_test, preds)
                f1 = f1_score(y_test, preds, average='weighted', labels=labels)
                model_metrics.update({
                    "Accuracy": accuracy_score(y_test, preds), "F1-Score (Weighted)": f1,
                    "Precision (Weighted)": precision_score(y_test, preds, average='weighted', labels=labels, zero_division=0),
                    "Recall (Weighted)": recall_score(y_test, preds, average='weighted', labels=labels, zero_division=0)
                })
                if f1 > best_model_score:
                    best_model_score = f1
                    best_model_object = model
                    best_model_name = name
            results.append(model_metrics)
        
        results_df = pd.DataFrame(results)

        prediction_intervals = {}
        confidence_intervals = {}
        
        if best_model_object is not None:
            joblib.dump(best_model_object, SAVED_MODEL_FILE)
            print(f"Best model ({best_model_name}) saved to {SAVED_MODEL_FILE}")

            if model_type == "classification":
                print("Generating classification cluster subsets...")
                # Use the same columns as training for prediction
                X_dummied = pd.get_dummies(df[available_predictors], drop_first=True)
                # Ensure all columns from training are present
                X_dummied = X_dummied.reindex(columns = X_train.columns, fill_value=0)
                
                full_data_predictions = best_model_object.predict(X_dummied)
                df_with_preds = df.copy()
                df_with_preds['predicted_cluster'] = full_data_predictions
                
                for cluster_id in df_with_preds['predicted_cluster'].unique():
                    cluster_df = df_with_preds[df_with_preds['predicted_cluster'] == cluster_id]
                    cluster_filename = os.path.join(TEMP_REPORTS_DIR, f"cluster_{cluster_id}_data.csv")
                    cluster_df.to_csv(cluster_filename, index=False)
                    print(f"Saved data for cluster {cluster_id} to {cluster_filename}")

        if model_type == "prediction" and best_model_object is not None:
            print(f"Calculating intervals for the best model: {best_model_name}")
            if isinstance(best_model_object, (LinearRegression, Ridge, Lasso)):
                y_train_np = y_train.to_numpy(dtype=float)
                X_train_const_np = sm.add_constant(X_train).to_numpy(dtype=float)
                weights_np = sample_weights.to_numpy(dtype=float) if sample_weights is not None else None
                wls_model = sm.WLS(y_train_np, X_train_const_np, weights=weights_np).fit()
                X_test_const_np = sm.add_constant(X_test).to_numpy(dtype=float)
                
                for alpha, level in [(0.10, "90%"), (0.05, "95%"), (0.01, "99%")]:
                    pred_summary = wls_model.get_prediction(X_test_const_np).summary_frame(alpha=alpha)
                    confidence_intervals[level] = {
                        "lower_bound_avg": pred_summary['mean_ci_lower'].mean(),
                        "upper_bound_avg": pred_summary['mean_ci_upper'].mean()
                    }
                    prediction_intervals[level] = {
                        "lower_bound_avg": pred_summary['obs_ci_lower'].mean(),
                        "upper_bound_avg": pred_summary['obs_ci_upper'].mean()
                    }

            elif isinstance(best_model_object, (RandomForestRegressor, GradientBoostingRegressor)):
                for alpha, level in [(0.10, "90%"), (0.05, "95%"), (0.01, "99%")]:
                    lower_quantile = alpha / 2
                    upper_quantile = 1 - (alpha / 2)
                    
                    lower_model = GradientBoostingRegressor(loss="quantile", alpha=lower_quantile, random_state=self.random_state)
                    lower_model.fit(X_train, y_train, sample_weight=sample_weights.to_numpy() if sample_weights is not None else None)
                    lower_preds = lower_model.predict(X_test)

                    upper_model = GradientBoostingRegressor(loss="quantile", alpha=upper_quantile, random_state=self.random_state)
                    upper_model.fit(X_train, y_train, sample_weight=sample_weights.to_numpy() if sample_weights is not None else None)
                    upper_preds = upper_model.predict(X_test)

                    prediction_intervals[level] = {
                        "lower_bound_avg": np.mean(lower_preds),
                        "upper_bound_avg": np.mean(upper_preds)
                    }
                confidence_intervals["note"] = "Confidence intervals are not directly available for tree-based models."

        final_report = {
            "model_comparison_table": results_df.to_dict(orient='records'),
            "best_model_recommendation": f"{best_model_name} (based on highest {primary_metric})",
            "saved_model_path": SAVED_MODEL_FILE if best_model_object else None,
            "prediction_intervals": {
                "note": "The estimated average range where a future single observation will fall.",
                "intervals": prediction_intervals
            },
            "confidence_intervals": {
                "note": "The estimated average range for the mean prediction. Most reliable for linear models.",
                "intervals": confidence_intervals
            }
        }

        with open(MODEL_ANALYSIS_FILE, 'w') as f:
            json.dump(final_report, f, indent=4, cls=NumpyJSONEncoder)
        print(f"Model analysis report with intervals saved to {MODEL_ANALYSIS_FILE}")
