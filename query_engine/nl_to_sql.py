import os
import re
import datetime
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from analytics.duckdb_engine import DuckDBAnalyticsEngine

# Try importing LangChain / Gemini LLM
try:
    from langchain_core.prompts import PromptTemplate
    from langchain_community.llms import FakeListLLM
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class NaturalLanguageQueryEngine:
    """
    LangChain & Rule-Based Natural Language to SQL Query Engine over DuckDB Parquet Data Lake.
    Converts user questions (e.g., 'From Jan to Feb till 10pm what was the flow?') into
    validated DuckDB SQL statements, executes them, and formats visual answers.
    """

    def __init__(self, duckdb_engine: DuckDBAnalyticsEngine, api_key: Optional[str] = None):
        self.engine = duckdb_engine
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.genai_client = None
        
        if self.api_key and GENAI_AVAILABLE:
            try:
                self.genai_client = genai.Client(api_key=self.api_key)
            except Exception:
                self.genai_client = None

    def query(self, question: str) -> Dict[str, Any]:
        """
        Main query entrypoint.
        Translates NL question -> SQL -> Executed DataFrame -> Explanation & Chart Type.
        """
        sql_query = None
        used_llm = False
        
        # 1. Try Gemini LLM if API Key is present
        if self.genai_client:
            try:
                sql_query = self._generate_sql_with_gemini(question)
                if sql_query:
                    used_llm = True
            except Exception:
                sql_query = None

        # 2. Fallback to Smart Pattern-Based SQL Generator
        if not sql_query:
            sql_query = self._generate_sql_with_rule_engine(question)

        # 3. Validate SQL Safety
        is_safe, error_msg = self._validate_sql(sql_query)
        if not is_safe:
            return {
                "success": False,
                "question": question,
                "sql": sql_query,
                "error": f"SQL Safety Validation Failed: {error_msg}",
                "result_df": None,
                "explanation": "Query contained disallowed SQL keywords.",
                "chart_type": None
            }

        # 4. Execute Query on DuckDB
        try:
            df = self.engine.execute_sql(sql_query)
            explanation, chart_type = self._synthesize_answer(question, sql_query, df)
            return {
                "success": True,
                "question": question,
                "sql": sql_query,
                "used_llm": used_llm,
                "result_df": df,
                "explanation": explanation,
                "chart_type": chart_type,
                "row_count": len(df)
            }
        except Exception as e:
            return {
                "success": False,
                "question": question,
                "sql": sql_query,
                "error": f"DuckDB Execution Error: {str(e)}",
                "result_df": None,
                "explanation": "Could not execute the generated SQL query.",
                "chart_type": None
            }

    def _validate_sql(self, sql: str) -> Tuple[bool, str]:
        """Ensures SQL is read-only SELECT statement."""
        if not sql:
            return False, "Empty SQL query generated."
        
        forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE TABLE", "TRUNCATE", "EXEC", "VACUUM"]
        sql_upper = sql.upper()
        for word in forbidden:
            if re.search(r'\b' + word + r'\b', sql_upper):
                return False, f"Forbidden command '{word}' detected."
                
        if not sql_upper.strip().startswith("SELECT") and not sql_upper.strip().startswith("WITH"):
            return False, "Query must start with SELECT or WITH."
            
        return True, ""

    def _generate_sql_with_gemini(self, question: str) -> str:
        """Generates SQL using Google Gemini API with schema context."""
        schema_info = self.engine.get_schema_metadata()
        prompt = f"""
You are an expert SQL engineer for an Industrial Data Lake using DuckDB over Parquet files.
Tables & Views available:
1. raw_sensor_data (timestamp, machine_id, sensor_id, sensor_type, location, value, unit, status, year, month, day, hour, date)
2. hourly_sensor_rollups (machine_id, sensor_id, sensor_type, location, unit, date, hour, avg_value, min_value, max_value, stddev_value, total_sum_value, record_count, anomaly_count)
3. daily_sensor_rollups (machine_id, sensor_id, sensor_type, location, unit, date, avg_value, min_value, max_value, stddev_value, total_sum_value, record_count, anomaly_count)
4. machine_health_summary (machine_id, total_sensors, total_readings, ok_readings, warning_readings, critical_readings, health_score_pct)

Sensor Types: 'flow_rate', 'temperature', 'pressure', 'vibration', 'power_consumption'.
Machines: 'PUMP-01', 'COMPRESSOR-02', 'TURBINE-03', 'CONVEYOR-04'.

User Question: "{question}"

Instructions:
- Write ONLY a valid DuckDB SQL query.
- Do NOT include markdown formatting or extra text outside ```sql blocks.
- For time-of-day filters (e.g. 'till 10pm'), use `hour <= 22` or `hour BETWEEN ...`.
- For date ranges (e.g. 'jan to feb'), use `date >= '2026-01-01' AND date <= '2026-02-28'`.
- Return only executable SQL text.
"""
        response = self.genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        text = response.text.strip()
        # Clean markdown wrappers
        text = re.sub(r'```sql\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        return text.strip()

    def _generate_sql_with_rule_engine(self, question: str) -> str:
        """
        Robust Rule-Based NL-to-SQL Translator.
        Handles complex industrial time-series questions including:
        - Date ranges ('jan to feb', 'january 10 to february 15')
        - Time of day ('till 10pm', 'after 8am', 'between 2pm and 6pm')
        - Sensor metrics ('flow', 'temperature', 'pressure', 'vibration', 'power')
        - Aggregations ('average', 'max', 'min', 'total', 'anomalies')
        - Machine filters ('PUMP-01', 'Compressor', 'Turbine')
        """
        q_lower = question.lower()
        
        # Determine Sensor Type
        sensor_type = None
        if "flow" in q_lower:
            sensor_type = "flow_rate"
        elif "temp" in q_lower or "heat" in q_lower:
            sensor_type = "temperature"
        elif "press" in q_lower:
            sensor_type = "pressure"
        elif "vibrat" in q_lower:
            sensor_type = "vibration"
        elif "power" in q_lower or "watt" in q_lower or "energy" in q_lower:
            sensor_type = "power_consumption"
            
        # Determine Machine ID
        machine_id = None
        if "pump" in q_lower or "pump-01" in q_lower:
            machine_id = "PUMP-01"
        elif "compressor" in q_lower or "compressor-02" in q_lower:
            machine_id = "COMPRESSOR-02"
        elif "turbine" in q_lower or "turbine-03" in q_lower:
            machine_id = "TURBINE-03"
        elif "conveyor" in q_lower or "conveyor-04" in q_lower:
            machine_id = "CONVEYOR-04"

        # Determine Date Range
        start_date = "2026-01-01"
        end_date = "2026-03-31"
        
        months = {
            "jan": ("2026-01-01", "2026-01-31"),
            "january": ("2026-01-01", "2026-01-31"),
            "feb": ("2026-02-01", "2026-02-28"),
            "february": ("2026-02-01", "2026-02-28"),
            "mar": ("2026-03-01", "2026-03-31"),
            "march": ("2026-03-01", "2026-03-31")
        }
        
        # Check 'from X to Y' or 'jan to feb'
        date_match = re.search(r'(jan|feb|mar|january|february|march)\s*(?:to|-|through)\s*(jan|feb|mar|january|february|march)', q_lower)
        if date_match:
            m1, m2 = date_match.group(1), date_match.group(2)
            if m1 in months and m2 in months:
                start_date = months[m1][0]
                end_date = months[m2][1]
        else:
            # Single month check
            for m_key, (s_d, e_d) in months.items():
                if m_key in q_lower:
                    start_date = s_d
                    end_date = e_d
                    break

        # Time of Day Filter (e.g. 'till 10pm', 'until 10 pm', 'before 8am')
        hour_clause = ""
        pm_match = re.search(r'(?:till|until|before)\s*(\d{1,2})\s*(pm|am)', q_lower)
        if pm_match:
            h_val = int(pm_match.group(1))
            is_pm = pm_match.group(2) == "pm"
            if is_pm and h_val < 12:
                h_val += 12
            elif not is_pm and h_val == 12:
                h_val = 0
            hour_clause = f" AND hour <= {h_val}"

        # Check Anomalies / Health Query
        if "anomaly" in q_lower or "anomalies" in q_lower or "critical" in q_lower or "warning" in q_lower or "spike" in q_lower:
            where_parts = [f"date >= '{start_date}' AND date <= '{end_date}'"]
            if sensor_type:
                where_parts.append(f"sensor_type = '{sensor_type}'")
            if machine_id:
                where_parts.append(f"machine_id = '{machine_id}'")
            where_parts.append("status IN ('WARNING', 'CRITICAL')")
            where_str = " AND ".join(where_parts)
            return f"SELECT timestamp, machine_id, sensor_id, sensor_type, value, unit, status FROM raw_sensor_data WHERE {where_str} ORDER BY value DESC LIMIT 20;"

        # Check Machine Health Summary
        if "health" in q_lower or "status summary" in q_lower:
            return "SELECT * FROM machine_health_summary ORDER BY health_score_pct DESC;"

        # Build Standard Time-Series Aggregation Query
        where_parts = [f"date >= '{start_date}' AND date <= '{end_date}'"]
        if hour_clause:
            where_parts.append(hour_clause.replace(" AND ", ""))
        if sensor_type:
            where_parts.append(f"sensor_type = '{sensor_type}'")
        if machine_id:
            where_parts.append(f"machine_id = '{machine_id}'")

        where_str = " AND ".join(where_parts)

        # Check if user asked for daily, hourly, or overall total/average
        if "daily" in q_lower:
            return f"""
SELECT date, 
       ROUND(AVG(avg_value), 2) as avg_value, 
       ROUND(SUM(total_sum_value), 2) as total_volume,
       ROUND(MAX(max_value), 2) as max_value,
       SUM(anomaly_count) as total_anomalies
FROM daily_sensor_rollups
WHERE {where_str}
GROUP BY date
ORDER BY date ASC;
""".strip()
        elif "hourly" in q_lower or hour_clause:
            return f"""
SELECT date, hour,
       ROUND(AVG(avg_value), 2) as avg_value, 
       ROUND(SUM(total_sum_value), 2) as total_volume,
       ROUND(MAX(max_value), 2) as max_value,
       SUM(anomaly_count) as total_anomalies
FROM hourly_sensor_rollups
WHERE {where_str}
GROUP BY date, hour
ORDER BY date ASC, hour ASC;
""".strip()
        else:
            return f"""
SELECT 
    COUNT(*) as reading_count,
    ROUND(AVG(value), 2) as avg_value,
    ROUND(MIN(value), 2) as min_value,
    ROUND(MAX(value), 2) as max_value,
    ROUND(SUM(value), 2) as total_cumulative_value,
    MAX(unit) as unit
FROM raw_sensor_data
WHERE {where_str};
""".strip()

    def _synthesize_answer(self, question: str, sql: str, df: pd.DataFrame) -> Tuple[str, str]:
        """
        Creates natural language answer summary & recommends chart visualization type.
        Chart Types: 'time_series', 'bar_chart', 'metric_card', 'table'
        """
        if df.empty:
            return "No matching records found in the industrial data lake for your criteria.", "table"

        cols = df.columns.tolist()

        # Case 1: Metric Card (single row with scalar metrics)
        if len(df) == 1 and ("avg_value" in cols or "reading_count" in cols or "health_score_pct" in cols):
            row = df.iloc[0]
            unit_str = f" {row.get('unit', '')}" if "unit" in row and pd.notnull(row["unit"]) else ""
            
            if "avg_value" in cols:
                avg_val = row["avg_value"]
                tot_val = row.get("total_cumulative_value", row.get("total_volume", None))
                tot_str = f" with a cumulative total of {tot_val:,.2f}{unit_str}" if tot_val is not None else ""
                text = f"Analyzed {row.get('reading_count', 'all'):,} records. The average value was **{avg_val}{unit_str}** (Min: {row.get('min_value', 'N/A')}, Max: {row.get('max_value', 'N/A')}){tot_str}."
                return text, "metric_card"

        # Case 2: Time Series Chart (has date / timestamp)
        if "date" in cols or "timestamp" in cols:
            row_cnt = len(df)
            first_date = df["date"].iloc[0] if "date" in cols else str(df["timestamp"].iloc[0])[:10]
            last_date = df["date"].iloc[-1] if "date" in cols else str(df["timestamp"].iloc[-1])[:10]
            
            metric_col = "avg_value" if "avg_value" in cols else ("value" if "value" in cols else cols[-1])
            max_row = df.loc[df[metric_col].idxmax()]
            
            text = f"Retrieved **{row_cnt} time-series entries** spanning from {first_date} to {last_date}. Peak recorded value was **{max_row[metric_col]}** on {max_row.get('date', max_row.get('timestamp', 'N/A'))}."
            return text, "time_series"

        # Case 3: Machine Health Bar Chart
        if "machine_id" in cols and "health_score_pct" in cols:
            text = f"Analyzed health scores across {len(df)} industrial machines. Highest operational health: **{df.iloc[0]['machine_id']}** ({df.iloc[0]['health_score_pct']}% OK)."
            return text, "bar_chart"

        # Default Table view
        return f"Successfully executed query and returned **{len(df):,} records**.", "table"
