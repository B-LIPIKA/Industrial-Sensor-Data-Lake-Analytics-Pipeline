import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from typing import Dict, Any, List

class DataLakeStorage:
    """
    Manages partitioned Parquet storage for the industrial sensor data lake.
    Organizes data by year=YYYY / month=MM / day=DD Hive partitioning.
    """

    def __init__(self, lake_dir: str = "data/lake"):
        self.lake_dir = lake_dir
        os.makedirs(self.lake_dir, exist_ok=True)

    def write_partitioned_parquet(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Transforms and writes DataFrame into Hive-partitioned Parquet files.
        Partitions by year, month, day.
        """
        if df.empty:
            return {"status": "empty", "records_written": 0, "partitions": []}

        df_work = df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_work["timestamp"]):
            df_work["timestamp"] = pd.to_datetime(df_work["timestamp"])
            
        # Extract partition keys
        df_work["year"] = df_work["timestamp"].dt.year
        df_work["month"] = df_work["timestamp"].dt.month.map(lambda m: f"{m:02d}")
        df_work["day"] = df_work["timestamp"].dt.day.map(lambda d: f"{d:02d}")
        df_work["hour"] = df_work["timestamp"].dt.hour
        df_work["date"] = df_work["timestamp"].dt.strftime("%Y-%m-%d")

        # Convert to PyArrow Table
        table = pa.Table.from_pandas(df_work)

        # Write to partitioned dataset
        pq.write_to_dataset(
            table=table,
            root_path=self.lake_dir,
            partition_cols=["year", "month", "day"],
            use_dictionary=True,
            compression="SNAPPY",
            existing_data_behavior="overwrite_or_ignore"
        )

        # Count partitions written
        unique_partitions = df_work[["year", "month", "day"]].drop_duplicates()
        partition_paths = [
            f"year={row['year']}/month={row['month']}/day={row['day']}"
            for _, row in unique_partitions.iterrows()
        ]

        return {
            "status": "success",
            "records_written": len(df_work),
            "partitions_updated": len(partition_paths),
            "sample_partitions": partition_paths[:5]
        }

    def list_all_parquet_files(self) -> List[str]:
        """Returns list of all Parquet file paths in data lake."""
        parquet_files = []
        for root, _, files in os.walk(self.lake_dir):
            for file in files:
                if file.endswith(".parquet"):
                    parquet_files.append(os.path.join(root, file))
        return parquet_files

    def get_storage_stats(self) -> Dict[str, Any]:
        """Returns total records, partition count, and size on disk."""
        files = self.list_all_parquet_files()
        total_size_bytes = sum(os.path.getsize(f) for f in files) if files else 0
        
        return {
            "file_count": len(files),
            "total_size_mb": round(total_size_bytes / (1024 * 1024), 2),
            "lake_directory": os.path.abspath(self.lake_dir)
        }
