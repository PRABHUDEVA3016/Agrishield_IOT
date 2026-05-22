import random
import numpy as np

# We import the crop profiles to align the simulator baseline with each crop
from src.data_generator import CROP_PROFILES

class IoTSimulator:
    """
    Simulates physical IoT sensor telemetries with realistic time-series drift,
    ambient-soil correlations, noise, and manual environmental scenarios.
    """
    def __init__(self):
        # Cache of previous sensor values to compute deltas
        self.last_sensors = {}

    def get_default_baseline(self, crop_name):
        """
        Creates a nominal baseline dictionary based on the selected crop profile.
        """
        if crop_name not in CROP_PROFILES:
            crop_name = "Wheat"
            
        profile = CROP_PROFILES[crop_name]
        
        # Calculate midpoints for ideal ranges
        m_min, m_max = profile["ideal_moisture"]
        ph_min, ph_max = profile["ideal_ph"]
        st_min, st_max = profile["ideal_soil_temp"]
        at_min, at_max = profile["ideal_ambient_temp"]
        h_min, h_max = profile["ideal_humidity"]
        
        return {
            "Soil_Moisture": float(np.mean([m_min, m_max])),
            "Soil_Temp": float(np.mean([st_min, st_max])),
            "pH": float(np.mean([ph_min, ph_max])),
            "MQ135_Gas": 180.0,  # normal clean air
            "Ambient_Temp": float(np.mean([at_min, at_max])),
            "Ambient_Humidity": float(np.mean([h_min, h_max])),
            "SW420_Vibration": 5.0,  # quiet
            "ESP32_Voltage": 3.42,
            "ESP32_RSSI": -55.0,
            "ESP32_Battery": 90.0
        }

    def generate_next_tick(self, current_sensors, crop_name, scenario="Normal"):
        """
        Simulates the next state (tick) of the IoT device based on the active crop
        and selected scenario. Adds random walk drift and noise.
        """
        # If empty, initialize baseline
        if not current_sensors:
            current_sensors = self.get_default_baseline(crop_name)
            
        profile = CROP_PROFILES[crop_name]
        next_state = current_sensors.copy()
        
        # Apply standard minor random walk drift (if scenario is Normal)
        if scenario == "Normal":
            # Small random drift
            next_state["Soil_Moisture"] += random.uniform(-0.5, 0.5)
            next_state["Soil_Temp"] += random.uniform(-0.2, 0.2)
            next_state["pH"] += random.uniform(-0.02, 0.02)
            next_state["MQ135_Gas"] += random.uniform(-2.0, 2.0)
            next_state["Ambient_Temp"] += random.uniform(-0.3, 0.3)
            next_state["Ambient_Humidity"] += random.uniform(-0.5, 0.5)
            next_state["SW420_Vibration"] = random.uniform(1.0, 8.0)
            
            # ESP32 battery slowly drains or stays relatively constant
            next_state["ESP32_Voltage"] += random.uniform(-0.002, 0.001)
            next_state["ESP32_RSSI"] += random.uniform(-1.0, 1.0)
            
        elif scenario == "Drought":
            # Rapid drying and heating
            next_state["Soil_Moisture"] -= random.uniform(1.5, 2.5)
            next_state["Soil_Temp"] += random.uniform(0.6, 1.2)
            next_state["pH"] += random.uniform(-0.01, 0.01)
            next_state["MQ135_Gas"] += random.uniform(0.5, 2.0)
            next_state["Ambient_Temp"] += random.uniform(0.8, 1.5)
            next_state["Ambient_Humidity"] -= random.uniform(1.0, 2.0)
            next_state["SW420_Vibration"] = random.uniform(2.0, 10.0)
            
            # Sun heating up the solar charging or thermal drain
            next_state["ESP32_Voltage"] += random.uniform(-0.005, 0.002)
            next_state["ESP32_RSSI"] += random.uniform(-0.5, 0.5)
            
        elif scenario == "Heavy Rain":
            # Instant waterlogging, cooling
            next_state["Soil_Moisture"] += random.uniform(2.0, 4.0)
            next_state["Soil_Temp"] -= random.uniform(0.4, 1.0)
            next_state["pH"] += random.uniform(-0.03, 0.01)  # Rain slightly acidifies
            next_state["MQ135_Gas"] -= random.uniform(1.0, 3.0)  # rain washes air
            next_state["Ambient_Temp"] -= random.uniform(0.5, 1.2)
            next_state["Ambient_Humidity"] += random.uniform(1.5, 3.5)
            next_state["SW420_Vibration"] = random.uniform(5.0, 15.0)  # rain vibration
            
            # Rain drops WiFi signal strength slightly
            next_state["ESP32_RSSI"] -= random.uniform(0.5, 2.0)
            next_state["ESP32_Voltage"] -= random.uniform(0.001, 0.005)

        elif scenario == "Animal Intrusion":
            # Moderate soil drift, but HUGE vibration spike
            next_state["Soil_Moisture"] += random.uniform(-0.1, 0.1)
            next_state["Soil_Temp"] += random.uniform(-0.1, 0.1)
            next_state["SW420_Vibration"] = random.uniform(65.0, 95.0)  # active intrusion!
            
            # ESP32 draws a bit more current for sending active warnings
            next_state["ESP32_Voltage"] -= random.uniform(0.008, 0.015)
            next_state["ESP32_RSSI"] += random.uniform(-3.0, 3.0)  # animal blocking line of sight
            
        elif scenario == "Compost Rotting":
            # Soil moisture and gas rise significantly
            next_state["Soil_Moisture"] += random.uniform(0.2, 0.8)
            next_state["Soil_Temp"] += random.uniform(0.3, 0.7)  # heat generated by compost rotting
            next_state["MQ135_Gas"] += random.uniform(15.0, 35.0)  # toxic NH3/CO2 released
            next_state["SW420_Vibration"] = random.uniform(1.0, 6.0)
            
            next_state["ESP32_Voltage"] += random.uniform(-0.002, 0.001)

        elif scenario == "Soil Acidification":
            # Rapid pH decrease (e.g. fertilizer run-off or contamination)
            next_state["pH"] -= random.uniform(0.15, 0.3)
            next_state["Soil_Moisture"] += random.uniform(-0.2, 0.2)
            next_state["Soil_Temp"] += random.uniform(-0.1, 0.1)
            next_state["MQ135_Gas"] += random.uniform(1.0, 5.0)
            
            next_state["ESP32_Voltage"] += random.uniform(-0.002, 0.001)
            
        # Hard bounds clipping to ensure physical realism
        next_state["Soil_Moisture"] = np.clip(next_state["Soil_Moisture"], 8.0, 98.0)
        next_state["Soil_Temp"] = np.clip(next_state["Soil_Temp"], 8.0, 52.0)
        next_state["pH"] = np.clip(next_state["pH"], 3.2, 10.8)
        next_state["MQ135_Gas"] = np.clip(next_state["MQ135_Gas"], 90.0, 990.0)
        next_state["Ambient_Temp"] = np.clip(next_state["Ambient_Temp"], 10.0, 50.0)
        next_state["Ambient_Humidity"] = np.clip(next_state["Ambient_Humidity"], 12.0, 98.0)
        next_state["SW420_Vibration"] = np.clip(next_state["SW420_Vibration"], 0.0, 100.0)
        next_state["ESP32_Voltage"] = np.clip(next_state["ESP32_Voltage"], 3.0, 3.6)
        next_state["ESP32_RSSI"] = np.clip(next_state["ESP32_RSSI"], -90.0, -30.0)
        
        # Calculate battery percentage directly from current voltage
        # ESP32 functions down to 3.0V (0%), full charge is 3.6V (100%)
        next_state["ESP32_Battery"] = np.clip((next_state["ESP32_Voltage"] - 3.0) / 0.6 * 100.0, 0.0, 100.0)
        
        # Cast everything as float
        for k, v in next_state.items():
            next_state[k] = float(v)
            
        return next_state

    def calculate_deltas(self, current_sensors, previous_sensors):
        """
        Calculates the differences between current and previous sensor readings
        to display trend indicators on the Streamlit dashboard metrics.
        """
        deltas = {}
        if not previous_sensors:
            for k in current_sensors.keys():
                deltas[k] = 0.0
            return deltas
            
        for k, v in current_sensors.items():
            prev_v = previous_sensors.get(k, v)
            deltas[k] = float(v - prev_v)
            
        return deltas
