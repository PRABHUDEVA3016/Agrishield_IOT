import os
import numpy as np
import pandas as pd

# Define Crop Profiles with their agronomic requirements
CROP_PROFILES = {
    "Ragi": {
        "ideal_moisture": (20.0, 45.0),
        "normal_moisture": (15.0, 55.0),
        "ideal_ph": (5.0, 8.2),
        "ideal_soil_temp": (22.0, 32.0),
        "ideal_ambient_temp": (20.0, 35.0),
        "ideal_humidity": (40.0, 70.0),
    },
    "Wheat": {
        "ideal_moisture": (30.0, 55.0),
        "normal_moisture": (25.0, 65.0),
        "ideal_ph": (6.0, 7.5),
        "ideal_soil_temp": (15.0, 25.0),
        "ideal_ambient_temp": (15.0, 28.0),
        "ideal_humidity": (50.0, 70.0),
    },
    "Rice": {
        "ideal_moisture": (65.0, 90.0),
        "normal_moisture": (55.0, 95.0),
        "ideal_ph": (5.5, 6.8),
        "ideal_soil_temp": (20.0, 35.0),
        "ideal_ambient_temp": (22.0, 38.0),
        "ideal_humidity": (70.0, 95.0),
    },
    "Sugarcane": {
        "ideal_moisture": (50.0, 75.0),
        "normal_moisture": (45.0, 80.0),
        "ideal_ph": (6.0, 7.5),
        "ideal_soil_temp": (25.0, 38.0),
        "ideal_ambient_temp": (25.0, 40.0),
        "ideal_humidity": (55.0, 85.0),
    },
    "Coconut": {
        "ideal_moisture": (40.0, 65.0),
        "normal_moisture": (30.0, 75.0),
        "ideal_ph": (5.2, 8.0),
        "ideal_soil_temp": (22.0, 32.0),
        "ideal_ambient_temp": (22.0, 35.0),
        "ideal_humidity": (60.0, 90.0),
    }
}

def generate_crop_data(crop_name, num_samples=1000, seed=None):
    """
    Generates realistic, physically-sound sensor data for a specific crop,
    mapping outputs dynamically to Soil Health, Crop Risk, and Irrigation targets.
    """
    if seed is not None:
        np.random.seed(seed)
        
    profile = CROP_PROFILES[crop_name]
    
    # 1. Base environmental features (with random variation & correlations)
    # Soil Moisture (DS18B20 & SEN0161)
    # Ranging from extremely dry (10%) to waterlogged (98%)
    moisture = np.random.uniform(10.0, 98.0, num_samples)
    
    # Soil Temp is inversely correlated with soil moisture (drier soil heats up faster)
    soil_temp_base = np.random.uniform(profile["ideal_soil_temp"][0] - 5, profile["ideal_soil_temp"][1] + 8, num_samples)
    soil_temp = soil_temp_base - 0.1 * (moisture - 40.0) + np.random.normal(0, 1.5, num_samples)
    soil_temp = np.clip(soil_temp, 8.0, 52.0)
    
    # pH Sensor (SEN0161) - centered around ideal with variance
    ph = np.random.normal(np.mean(profile["ideal_ph"]), 1.2, num_samples)
    ph = np.clip(ph, 3.5, 10.5)
    
    # MQ-135 Gas Sensor - usually low (100-300 ppm) but occasional high values (400-900) representing decomposition, smoke or fertilizer buildup
    gas_base = np.random.exponential(scale=150.0, size=num_samples) + 120.0
    # Higher gas when soil is extremely wet (anaerobic rotting) or extremely hot
    gas = gas_base + 1.5 * (soil_temp - 25.0) + 2.0 * np.maximum(0, moisture - 75.0)
    gas = np.clip(gas, 100.0, 990.0)
    
    # DHT11 Ambient Temperature & Ambient Humidity
    ambient_temp = soil_temp + np.random.normal(2.0, 2.0, num_samples)
    ambient_temp = np.clip(ambient_temp, 12.0, 48.0)
    
    # Humidity is negatively correlated with ambient temperature, positively with soil moisture
    humidity = 95.0 - 1.2 * (ambient_temp - 20.0) + 0.15 * moisture + np.random.normal(0, 5.0, num_samples)
    humidity = np.clip(humidity, 15.0, 98.0)
    
    # SW-420 Vibration Sensor - mostly quiet (<15), occasional spikes representing animals, farming activity
    vibration = np.random.choice([0.0, 1.0], size=num_samples, p=[0.85, 0.15]) * np.random.uniform(20, 95, num_samples)
    vibration += np.random.uniform(1.0, 12.0, num_samples)
    
    # ESP32-WROOM-32 internal metrics
    esp_voltage = np.random.uniform(3.2, 3.58, num_samples)
    # Drop voltage slightly if WiFi signal is weak or vibration spikes (esp transmitting active alerts)
    esp_voltage -= 0.001 * vibration
    esp_rssi = np.random.uniform(-85.0, -40.0, num_samples)
    esp_battery = np.clip((esp_voltage - 3.0) / 0.6 * 100.0, 0.0, 100.0)
    
    # 2. Rule-Based Targets (Soil Health, Crop Risk, Irrigation Recommendation)
    soil_health = []
    crop_risk = []
    irrigation_req = []
    
    for i in range(num_samples):
        m = moisture[i]
        t_s = soil_temp[i]
        p = ph[i]
        g = gas[i]
        t_a = ambient_temp[i]
        h_a = humidity[i]
        vib = vibration[i]
        
        # A. Soil Health Status: Excellent (0), Good (1), Degraded (2)
        # Optimal parameters
        ph_ideal_min, ph_ideal_max = profile["ideal_ph"]
        m_ideal_min, m_ideal_max = profile["ideal_moisture"]
        
        # Acidic/Alkaline penalty
        ph_dev = 0
        if p < ph_ideal_min or p > ph_ideal_max:
            ph_dev = min(abs(p - ph_ideal_min), abs(p - ph_ideal_max))
            
        # Moisture penalty
        moist_dev = 0
        if m < m_ideal_min or m > m_ideal_max:
            moist_dev = min(abs(m - m_ideal_min), abs(m - m_ideal_max))
            
        if ph_dev < 0.4 and moist_dev < 10.0 and g < 320.0:
            health = "Excellent"
        elif ph_dev < 1.2 and moist_dev < 25.0 and g < 550.0:
            health = "Good"
        else:
            health = "Degraded"
        soil_health.append(health)
        
        # B. Crop Risk Level: Low (0), Medium (1), High (2)
        # High crop risk triggers:
        # - Extreme water stress (drought or flooding based on crop)
        # - Toxic gasses (compost rotting/smoke)
        # - Severe soil degradation
        # - Wild animal intrusion (vibration > 60)
        # - Temperature stress
        
        m_norm_min, m_norm_max = profile["normal_moisture"]
        
        is_drought = m < m_norm_min - 5.0
        is_waterlogged = m > m_norm_max + 8.0 if crop_name != "Rice" else m > 96.0  # Rice likes water
        is_intrusion = vib > 55.0
        is_acid_base_extreme = p < 4.6 or p > 9.2
        is_heat_stress = t_s > (profile["ideal_soil_temp"][1] + 6.0)
        
        # Moderate deviations
        is_mod_drought = (m < m_ideal_min) and not is_drought
        is_mod_wet = (m > m_ideal_max) and not is_waterlogged
        is_mod_intrusion = 25.0 < vib <= 55.0
        is_mod_temp = (t_s > profile["ideal_soil_temp"][1]) and not is_heat_stress
        
        if is_drought or is_waterlogged or is_intrusion or is_acid_base_extreme or is_heat_stress:
            risk = "High"
        elif is_mod_drought or is_mod_wet or is_mod_intrusion or is_mod_temp or health == "Degraded" or g > 450.0:
            risk = "Medium"
        else:
            risk = "Low"
        crop_risk.append(risk)
        
        # C. Irrigation Recommendation: No Irrigation (0), Light Irrigation (1), Heavy Irrigation (2)
        # Rice requires waterlogged conditions, Ragi needs very dry soil, etc.
        if m >= m_ideal_min:
            irr = "No Irrigation"
        else:
            # If moisture is below ideal lower bound
            crit_threshold = m_ideal_min - 12.0
            if m < crit_threshold or (m < m_ideal_min - 5.0 and t_a > 36.0):
                irr = "Heavy Irrigation"
            else:
                irr = "Light Irrigation"
        irrigation_req.append(irr)
        
    df = pd.DataFrame({
        "Crop_Type": crop_name,
        "Soil_Moisture": moisture,
        "Soil_Temp": soil_temp,
        "pH": ph,
        "MQ135_Gas": gas,
        "Ambient_Temp": ambient_temp,
        "Ambient_Humidity": humidity,
        "SW420_Vibration": vibration,
        "ESP32_Voltage": esp_voltage,
        "ESP32_RSSI": esp_rssi,
        "ESP32_Battery": esp_battery,
        "Soil_Health": soil_health,
        "Crop_Risk": crop_risk,
        "Irrigation_Req": irrigation_req
    })
    
    return df

def generate_complete_dataset(samples_per_crop=1000, output_dir=None):
    """
    Generates a full agricultural dataset featuring all 5 target crops,
    combines them, and exports them to a CSV.
    """
    dfs = []
    # Use standard seeds for reproducibility of the initial dataset
    for i, crop in enumerate(CROP_PROFILES.keys()):
        df_crop = generate_crop_data(crop, num_samples=samples_per_crop, seed=42 + i)
        dfs.append(df_crop)
        
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # Shuffle the dataset
    df_combined = df_combined.sample(frac=1.0, random_state=100).reset_index(drop=True)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "precision_agriculture_data.csv")
        df_combined.to_csv(csv_path, index=False)
        print(f"Dataset generated successfully at {csv_path} with {len(df_combined)} rows.")
        return csv_path
    
    return df_combined

if __name__ == "__main__":
    # Generate directly if executed
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(workspace_dir, "data")
    generate_complete_dataset(samples_per_crop=1000, output_dir=data_dir)
