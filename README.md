# AgriShield IoT™: AI-Powered Precision Agriculture System 🌾🇮🇳

**AgriShield IoT** is a complete, startup-ready decision support prototype designed for small-scale Indian farmers using low-cost smart farming technology. It integrates real-time IoT hardware simulation, crop-aware physical-biochemical rule engines, and three distinct Machine Learning Classifiers (Random Forest) to protect crops, optimize watering schedules, and monitor soil biology in real time.

This project is tailored for agricultural hackathons, agritech startup pitch competitions, and engineering demonstrations.

---

## 🚀 Key Features

1. **Vibrant Glassmorphic Dashboard (Streamlit)**: Designed with a modern, professional green dark/light interface, custom CSS cards, and responsive data visualizations.
2. **Crop-Aware AI Engine**: Tailored for regional crops: **Ragi (Finger Millet), Wheat, Rice, Sugarcane, and Coconut**. 
   - Predictions are *crop-specific*! For instance, $30\%$ soil moisture represents a healthy growth state for drought-tolerant Ragi, but represents a critical risk requiring heavy irrigation for Rice.
3. **Multi-Target Machine Learning**: Evaluates and predicts three separate critical factors:
   - **Soil Health Status**: Classifies soil as `Excellent`, `Good`, or `Degraded`.
   - **Crop Risk Level**: Detects thermal, hydration, chemical, or security stress as `Low`, `Medium`, or `High`.
   - **Precision Irrigation Advisor**: Directs farmers on watering actions: `No Irrigation`, `Light Irrigation`, or `Heavy Irrigation`.
4. **Active Environmental Presets**: Sidebar controls allow immediate simulation of complex environmental events (e.g. Drought Stress, Monsoon Waterlogging, Wild Animal Intrusion, Compost Decay, or Soil Acidification).
5. **Real-time SMS & Deterrent Alerts**: High-fidelity security warnings for vibration spikes (such as animal crossing or trespassers) and chemical degradation.
6. **Developer/Agro-Analyst Workshop**: Interactive live-training zone allowing adjustments of hyperparameters (estimators, max depth, test split) and instant evaluation of accuracies, MDI feature importances, and confusion matrices.

---

## 🔌 Hardware Sensor Suite Simulated
The system models low-cost, off-the-shelf components that small-scale farmers can easily deploy:

*   **Soil Moisture Sensor (Analog)**: Measures soil volumetric water content ($10\% - 98\%$).
*   **DS18B20 Soil Temperature Sensor**: High precision waterproof probe measuring sub-surface temperature ($10^\circ\text{C} - 52^\circ\text{C}$).
*   **SEN0161 pH Sensor**: Measures soil acidity or alkalinity ($3.5 - 10.5$) to gauge nutrient lock-ins.
*   **MQ-135 Gas Sensor**: Monitors ambient organic decay, toxic gases (NH3, CO2), or crop-burning smoke ($100 - 1000$ ppm).
*   **DHT11 Temperature & Humidity Sensor**: Measures micro-climate ambient temp ($12^\circ\text{C} - 48^\circ\text{C}$) and ambient humidity ($15\% - 98\%$).
*   **SW-420 Vibration Sensor**: Binary/continuous vibration trigger used as a security fence guard to detect animal crop raiding or machinery trespassing.
*   **ESP32-WROOM-32 (Microcontroller)**: Core telemetry node; reports system voltage ($3.0\text{V} - 3.6\text{V}$), solar battery charge ($0\% - 100\%$), and WiFi signal strength RSSI ($-90 - -30$ dBm).

---

## 📊 Regional Crop Parameters

The machine learning models are trained on $5,000$ synthetic observations mapped strictly to regional Indian crop characteristics:

| Crop | Ideal Soil Moisture | Ideal pH | Ideal Soil Temp | Ideal Ambient Humidity | Drought / Waterlogging Tolerance |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Ragi** | $20\% - 45\%$ | $5.0 - 8.2$ | $22^\circ\text{C} - 32^\circ\text{C}$ | $40\% - 70\%$ | High drought tolerance; extremely sensitive to waterlogging. |
| **Wheat** | $30\% - 55\%$ | $6.0 - 7.5$ | $15^\circ\text{C} - 25^\circ\text{C}$ | $50\% - 70\%$ | Prefers cool winter soil; sensitive to extreme dry soils. |
| **Rice (Paddy)** | $65\% - 90\%$ | $5.5 - 6.8$ | $20^\circ\text{C} - 35^\circ\text{C}$ | $70\% - 95\%$ | Waterlogged fields required; highly sensitive to dry conditions. |
| **Sugarcane** | $50\% - 75\%$ | $6.0 - 7.5$ | $25^\circ\text{C} - 38^\circ\text{C}$ | $55\% - 85\%$ | Warm soil loving; needs continuous high irrigation cycles. |
| **Coconut** | $40\% - 65\%$ | $5.2 - 8.0$ | $22^\circ\text{C} - 32^\circ\text{C}$ | $60\% - 90\%$ | Sandy soil and salinity adaptable; requires high ambient humidity. |

---

## 🛠️ Step-by-Step Installation

### Prerequisites
Make sure you have **Python 3.8+** installed on your system.

### 1. Navigate to the Subdirectory
Open your terminal and navigate to the project directory:
```bash
cd agri_shield_iot
```

### 2. Create a Virtual Environment (Recommended)
Creating a virtual environment ensures clean dependencies:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Run Verification (Optional)
To verify the dataset generation, ML training pipeline, and crop-aware scoring manually:
```bash
python verify_agri_shield.py
```

### 5. Launch the Streamlit Dashboard
Execute the application using Streamlit:
```bash
streamlit run app.py
```

---

## 🧠 Machine Learning Engine Details

*   **Algorithms**: Three distinct **Random Forest Classifiers** fitted on preprocessed and scaled inputs.
*   **Preprocessing**: 
    - Numerical features are standardized using `StandardScaler` to bring soil telemetry and air metrics onto a common scale.
    - Categorical `Crop_Type` is manual one-hot encoded to maintain robust column alignment across training and live hardware inference.
*   **Pickle/Joblib Packaging**: All parameters (fitted scaler, feature lists, trained models, and validation metrics) are combined into a serialized binary payload at `models/precision_agri_pipeline.joblib`.
*   **Evaluation Outputs**:
    - **Interactive Plotly Confusion Matrix**: Renders true vs. predicted classifications to visualize false positive irrigation commands or risk alarms.
    - **Feature Importance**: Uses Mean Decrease in Impurity (MDI) to map which sensor triggers are driving each decision. (e.g., Soil moisture is highly crucial for irrigation, while SW-420 vibration dominates crop risk).

---

## 🚜 Startup Team & Startup Showcase Guide

### Pitching Tips for Competitions:
1. **The Problem**: Over $70\%$ of Indian smallholders lack access to soil testing labs and fail to gauge microclimate events or security trespassers. 
2. **The Solution**: AgriShield IoT packages $500$ rupees worth of off-the-shelf sub-sensors with an ESP32 and runs light local machine learning to trigger localized security sirens and automated solenoid drip irrigation.
3. **Demo Strategy**: 
   - Open **Tab 1** and select **Rice** with the scenario **Normal**. Show that the system is stable.
   - Switch the scenario to **Drought**. Click **Fetch Next Telemetry Tick** and show how the system detects **High Risk**, triggers **Heavy Irrigation**, and alerts the farmer immediately.
   - Switch the scenario to **Animal Intrusion** and watch the **Vibration security alert** activate instantly!
   - Navigate to **Tab 3** to show the judges the model accuracies, confusion matrices, and feature importance to prove the validity of your startup's AI engine.
