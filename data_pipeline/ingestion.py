import os
import pandas as pd
from typing import Dict, Any, Tuple
from data_pipeline.quality import DataQualityEngine
from data_pipeline.storage import DataLakeStorage

class IngestionPipeline:
    """
    End-to-End Ingestion Pipeline for Industrial Sensor Telemetry.
    Integrates incremental ingestion, quality audit, transformation, and Parquet lake writing.
    """
    
    def __init__(self, lake_dir: str = "data/lake"):
        self.storage = DataLakeStorage(lake_dir=lake_dir)

    def process_file_or_dataframe(
        self,
        input_source: Any
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Processes a CSV file path, uploaded file, or pandas DataFrame.
        """
        if isinstance(input_source, str):
            if input_source.endswith(".csv"):
                df = pd.read_csv(input_source)
            elif input_source.endswith(".parquet"):
                df = pd.read_parquet(input_source)
            else:
                raise ValueError("Unsupported file type. Use CSV or Parquet.")
        elif isinstance(input_source, pd.DataFrame):
            df = input_source
        elif hasattr(input_source, "name"): # Streamlit UploadedFile
            if input_source.name.endswith(".csv"):
                df = pd.read_csv(input_source)
            elif input_source.name.endswith(".parquet"):
                df = pd.read_parquet(input_source)
            else:
                raise ValueError("Uploaded file must be CSV or Parquet.")
        else:
            raise ValueError("Invalid input source.")

        # 1. Quality Audit
        passed, quality_report, cleaned_df = DataQualityEngine.validate_dataset(df)
        
        if not passed:
            return False, {
                "stage": "quality_validation_failed",
                "quality_report": quality_report,
                "storage_summary": None
            }
            
        # 2. Storage Partitioning
        storage_summary = self.storage.write_partitioned_parquet(cleaned_df)
        
        return True, {
            "stage": "completed",
            "quality_report": quality_report,
            "storage_summary": storage_summary,
            "processed_records": len(cleaned_df)
        }
