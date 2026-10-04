import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_industrial_sensor_dataset(
    start_date: str = "2026-01-01",
    end_date: str = "2026-03-31",
    freq: str = "1h",
    output_path: str = "data/sample/industrial_sensor_data.csv"
) -> pd.DataFrame:
    """
    Generates a realistic industrial sensor telemetry dataset.
    Includes flow rate, temperature, pressure, vibration, and power consumption
    with daily/hourly diurnal patterns and synthetic anomalies.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    machines = [
        {"id": "PUMP-01", "location": "Facility_Alpha_Zone_1"},
        {"id": "COMPRESSOR-02", "location": "Facility_Alpha_Zone_2"},
        {"id": "TURBINE-03", "location": "Facility_Beta_Zone_1"},
        {"id": "CONVEYOR-04", "location": "Facility_Beta_Zone_2"}
    ]
    
    sensor_specs = {
        "flow_rate": {"unit": "L/min", "base": 250.0, "noise": 15.0, "min": 50.0, "max": 450.0, "anomaly_high": 420.0},
        "temperature": {"unit": "°C", "base": 65.0, "noise": 4.0, "min": 20.0, "max": 120.0, "anomaly_high": 95.0},
        "pressure": {"unit": "bar", "base": 6.5, "noise": 0.5, "min": 1.0, "max": 15.0, "anomaly_high": 12.0},
        "vibration": {"unit": "mm/s", "base": 2.1, "noise": 0.3, "min": 0.1, "max": 10.0, "anomaly_high": 6.5},
        "power_consumption": {"unit": "kW", "base": 45.0, "noise": 5.0, "min": 5.0, "max": 100.0, "anomaly_high": 85.0}
    }
    
    records = []
    
    np.random.seed(42)
    
    for machine in machines:
        m_id = machine["id"]
        location = machine["location"]
        
        for sensor_type, spec in sensor_specs.items():
            sensor_id = f"SNS-{m_id[:3]}-{sensor_type[:4].upper()}"
            
            base_val = spec["base"]
            noise_std = spec["noise"]
            unit = spec["unit"]
            
            for ts in timestamps:
                # Add diurnal cycle (day vs night variation)
                hour = ts.hour
                diurnal_shift = np.sin((hour - 6) * np.pi / 12) * (base_val * 0.1)
                
                # Random noise
                val = base_val + diurnal_shift + np.random.normal(0, noise_std)
                
                # Introduce occasional anomaly (1% chance)
                is_anomaly = np.random.rand() < 0.015
                if is_anomaly:
                    val += np.random.choice([1, -1]) * (base_val * 0.6)
                
                # Clamp value
                val = float(np.clip(val, spec["min"], spec["max"]))
                
                # Determine status
                if val >= spec["anomaly_high"]:
                    status = "CRITICAL" if val >= spec["anomaly_high"] * 1.15 else "WARNING"
                else:
                    status = "OK"
                
                records.append({
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "machine_id": m_id,
                    "sensor_id": sensor_id,
                    "sensor_type": sensor_type,
                    "location": location,
                    "value": round(val, 2),
                    "unit": unit,
                    "status": status
                })
                
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df):,} sensor records saved to {output_path}")
    return df

if __name__ == "__main__":
    generate_industrial_sensor_dataset()
