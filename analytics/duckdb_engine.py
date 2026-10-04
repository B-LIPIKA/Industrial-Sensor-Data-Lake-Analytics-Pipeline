import os
import duckdb
import pandas as pd
from typing import Dict, Any, List, Optional

class DuckDBAnalyticsEngine:
    """
    DuckDB-based analytical layer enabling ultra-fast SQL queries directly over
    partitioned Parquet files and automated hourly/daily rollups.
    """

    def __init__(self, lake_dir: str = "data/lake", db_path: str = ":memory:"):
        self.lake_dir = os.path.abspath(lake_dir)
        self.con = duckdb.connect(database=db_path)
        self.refresh_views()

    def refresh_views(self):
        """
        Creates or updates DuckDB analytical views over partitioned Parquet files.
        """
        parquet_glob = os.path.join(self.lake_dir, "**", "*.parquet").replace("\\", "/")
        
        # Check if Parquet files exist
        has_files = False
        for root, _, files in os.walk(self.lake_dir):
            if any(f.endswith(".parquet") for f in files):
                has_files = True
                break

        if not has_files:
            # Create empty dummy table schema if no parquet files yet
            self.con.execute("""
                CREATE OR REPLACE VIEW raw_sensor_data AS
                SELECT 
                    CAST('2026-01-01 00:00:00' AS TIMESTAMP) as timestamp,
                    'DUMMY' as machine_id,
                    'SNS-DUMMY' as sensor_id,
                    'temperature' as sensor_type,
                    'Zone_1' as location,
                    0.0 as value,
                    '°C' as unit,
                    'OK' as status,
                    2026 as year,
                    '01' as month,
                    '01' as day,
                    0 as hour,
                    '2026-01-01' as date,
                    true as is_valid_range,
                    100.0 as quality_score
                WHERE 1=0;
            """)
        else:
            self.con.execute(f"""
                CREATE OR REPLACE VIEW raw_sensor_data AS 
                SELECT *,
                    CAST(timestamp AS TIMESTAMP) as timestamp_ts
                FROM read_parquet('{parquet_glob}', hive_partitioning=1);
            """)

        # 1. Hourly Rollups View
        self.con.execute("""
            CREATE OR REPLACE VIEW hourly_sensor_rollups AS
            SELECT 
                machine_id,
                sensor_id,
                sensor_type,
                location,
                unit,
                date,
                hour,
                AVG(value) as avg_value,
                MIN(value) as min_value,
                MAX(value) as max_value,
                STDDEV_SAMP(value) as stddev_value,
                SUM(value) as total_sum_value,
                COUNT(*) as record_count,
                COUNT(CASE WHEN status IN ('WARNING', 'CRITICAL') THEN 1 END) as anomaly_count
            FROM raw_sensor_data
            GROUP BY machine_id, sensor_id, sensor_type, location, unit, date, hour;
        """)

        # 2. Daily Rollups View
        self.con.execute("""
            CREATE OR REPLACE VIEW daily_sensor_rollups AS
            SELECT 
                machine_id,
                sensor_id,
                sensor_type,
                location,
                unit,
                date,
                AVG(value) as avg_value,
                MIN(value) as min_value,
                MAX(value) as max_value,
                STDDEV_SAMP(value) as stddev_value,
                SUM(value) as total_sum_value,
                COUNT(*) as record_count,
                COUNT(CASE WHEN status IN ('WARNING', 'CRITICAL') THEN 1 END) as anomaly_count
            FROM raw_sensor_data
            GROUP BY machine_id, sensor_id, sensor_type, location, unit, date;
        """)

        # 3. Machine Health Summary View
        self.con.execute("""
            CREATE OR REPLACE VIEW machine_health_summary AS
            SELECT 
                machine_id,
                COUNT(DISTINCT sensor_id) as total_sensors,
                COUNT(*) as total_readings,
                COUNT(CASE WHEN status = 'OK' THEN 1 END) as ok_readings,
                COUNT(CASE WHEN status = 'WARNING' THEN 1 END) as warning_readings,
                COUNT(CASE WHEN status = 'CRITICAL' THEN 1 END) as critical_readings,
                ROUND((COUNT(CASE WHEN status = 'OK' THEN 1 END)::FLOAT / COUNT(*)) * 100, 2) as health_score_pct
            FROM raw_sensor_data
            GROUP BY machine_id;
        """)

    def execute_sql(self, sql_query: str) -> pd.DataFrame:
        """
        Executes a SQL query against the DuckDB analytical engine.
        Returns result as pandas DataFrame.
        """
        self.refresh_views()
        return self.con.execute(sql_query).fetchdf()

    def get_kpi_summary(self) -> Dict[str, Any]:
        """Returns overall data lake high-level KPIs."""
        self.refresh_views()
        df_count = self.con.execute("SELECT COUNT(*) as cnt FROM raw_sensor_data").fetchdf()
        total_records = int(df_count["cnt"].iloc[0]) if not df_count.empty else 0
        
        if total_records == 0:
            return {
                "total_records": 0,
                "machines_monitored": 0,
                "date_range": ("N/A", "N/A"),
                "total_anomalies": 0,
                "overall_health_pct": 100.0
            }

        stats = self.con.execute("""
            SELECT 
                COUNT(DISTINCT machine_id) as machines,
                MIN(date) as min_date,
                MAX(date) as max_date,
                COUNT(CASE WHEN status IN ('WARNING', 'CRITICAL') THEN 1 END) as anomalies
            FROM raw_sensor_data
        """).fetchdf().iloc[0]

        health_pct = round(100.0 * (1 - (stats["anomalies"] / total_records)), 2)

        return {
            "total_records": total_records,
            "machines_monitored": int(stats["machines"]),
            "date_range": (str(stats["min_date"]), str(stats["max_date"])),
            "total_anomalies": int(stats["anomalies"]),
            "overall_health_pct": health_pct
        }

    def get_schema_metadata(self) -> Dict[str, List[str]]:
        """Returns list of tables/views and their column names."""
        tables = ["raw_sensor_data", "hourly_sensor_rollups", "daily_sensor_rollups", "machine_health_summary"]
        metadata = {}
        for t in tables:
            cols = self.con.execute(f"DESCRIBE {t}").fetchdf()["column_name"].tolist()
            metadata[t] = cols
        return metadata
