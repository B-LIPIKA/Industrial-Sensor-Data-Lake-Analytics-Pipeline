import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

class DataQualityEngine:
    """
    Data-Quality Validation Engine for Industrial Sensor Telemetry.
    Executes automated data health checks, schema enforcement, missing value audits,
    and out-of-range value detection.
    """
    
    REQUIRED_COLUMNS = ["timestamp", "machine_id", "sensor_id", "sensor_type", "value"]
    
    SENSOR_RANGES = {
        "flow_rate": (0.0, 1000.0),
        "temperature": (-50.0, 250.0),
        "pressure": (0.0, 50.0),
        "vibration": (0.0, 50.0),
        "power_consumption": (0.0, 1000.0)
    }

    @classmethod
    def validate_dataset(cls, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any], pd.DataFrame]:
        """
        Validates an incoming dataset dataframe.
        Returns:
            passed (bool): Whether dataset passes critical quality checks.
            report (dict): Detailed summary metrics of quality inspection.
            cleaned_df (DataFrame): DataFrame with sanitized data types & quality flags.
        """
        report = {
            "total_records": len(df),
            "missing_values": {},
            "out_of_range_count": 0,
            "duplicate_count": 0,
            "schema_valid": True,
            "missing_columns": [],
            "quality_score": 100.0
        }
        
        cleaned_df = df.copy()
        
        # 1. Schema Validation
        missing_cols = [col for col in cls.REQUIRED_COLUMNS if col not in cleaned_df.columns]
        if missing_cols:
            report["schema_valid"] = False
            report["missing_columns"] = missing_cols
            report["quality_score"] -= 40.0
            return False, report, cleaned_df
        
        # 2. Timestamp Parsing
        try:
            cleaned_df["timestamp"] = pd.to_datetime(cleaned_df["timestamp"])
        except Exception as e:
            report["schema_valid"] = False
            report["timestamp_error"] = str(e)
            report["quality_score"] -= 30.0
            return False, report, cleaned_df
        
        # 3. Missing Value Audit
        total_cells = len(cleaned_df) * len(cls.REQUIRED_COLUMNS)
        missing_cells = 0
        for col in cls.REQUIRED_COLUMNS:
            null_cnt = int(cleaned_df[col].isnull().sum())
            report["missing_values"][col] = null_cnt
            missing_cells += null_cnt
            
        if missing_cells > 0:
            missing_pct = (missing_cells / total_cells) * 100
            report["quality_score"] -= min(30.0, missing_pct * 2)
            # Impute or drop missing
            cleaned_df = cleaned_df.dropna(subset=cls.REQUIRED_COLUMNS)

        # 4. Duplicate Check
        duplicate_mask = cleaned_df.duplicated(subset=["timestamp", "machine_id", "sensor_id"])
        dup_count = int(duplicate_mask.sum())
        report["duplicate_count"] = dup_count
        if dup_count > 0:
            report["quality_score"] -= min(15.0, (dup_count / len(cleaned_df)) * 100)
            cleaned_df = cleaned_df.drop_duplicates(subset=["timestamp", "machine_id", "sensor_id"])

        # 5. Sensor Value Range Audit
        cleaned_df["value"] = pd.to_numeric(cleaned_df["value"], errors="coerce")
        out_of_range_mask = pd.Series(False, index=cleaned_df.index)
        
        for stype, (min_v, max_v) in cls.SENSOR_RANGES.items():
            type_mask = (cleaned_df["sensor_type"] == stype)
            range_err = type_mask & ((cleaned_df["value"] < min_v) | (cleaned_df["value"] > max_v))
            out_of_range_mask = out_of_range_mask | range_err
            
        oor_count = int(out_of_range_mask.sum())
        report["out_of_range_count"] = oor_count
        if oor_count > 0:
            report["quality_score"] -= min(20.0, (oor_count / len(cleaned_df)) * 100)
            
        # Add quality flags to cleaned_df
        cleaned_df["is_valid_range"] = ~out_of_range_mask
        cleaned_df["quality_score"] = max(0.0, round(report["quality_score"], 2))
        
        passed = report["quality_score"] >= 70.0
        return passed, report, cleaned_df
