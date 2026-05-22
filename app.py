import os
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pyrebase

# Set Streamlit Page Configuration with startup icon
st.set_page_config(
    page_title="AgriShield IoT - AI Precision Agriculture",
    page_icon="🚜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Vibrant Green & Glassmorphism Design System)
st.markdown("""
<style>
    /* Main body background and font styling */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f0f7f4 0%, #e6f3ed 100%);
    }
    
    /* Header styling */
   
    .app-header {
    background: linear-gradient(90deg, #1b5e20 0%, #2e7d32 100%);
    color: white;
    padding: 1rem 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.2rem;
    box-shadow: 0 3px 10px rgba(46, 125, 50, 0.12);
}

.app-header h1 {
    margin: 0;
    font-weight: 700;
    font-size: 2rem;
}

.app-header p {
    margin-top: 4px;
    font-size: 0.95rem;
    color: #e8f5e9;
}
    /* Card design elements */
    .glass-card {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.05);
        margin-bottom: 1.5rem;
    }
    
    /* Metric container styling */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid #c8e6c9 !important;
        border-radius: 16px !important;
        padding: 1rem 1.2rem !important;
        box-shadow: 0 4px 15px rgba(46, 125, 50, 0.04) !important;
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 6px 20px rgba(46, 125, 50, 0.08) !important;
        border-color: #81c784 !important;
    }
    
    /* Prediction Cards badge colors */
    .prediction-badge {
        font-weight: 700;
        font-size: 1.1rem;
        padding: 6px 12px;
        border-radius: 20px;
        display: inline-block;
        margin-top: 5px;
    }
    
    .badge-excellent { background-color: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; }
    .badge-good { background-color: #eef2f7; color: #1565c0; border: 1px solid #d0e1f9; }
    .badge-degraded { background-color: #ffebee; color: #c62828; border: 1px solid #ffcdd2; }
    .badge-low { background-color: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; }
    .badge-medium { background-color: #fff8e1; color: #ef6c00; border: 1px solid #ffe082; }
    .badge-high { background-color: #ffebee; color: #c62828; border: 1px solid #ffcdd2; }
    .badge-irr-none { background-color: #e3f2fd; color: #0d47a1; border: 1px solid #bbdefb; }
    .badge-irr-light { background-color: #e0f7fa; color: #006064; border: 1px solid #b2ebf2; }
    .badge-irr-heavy { background-color: #e1f5fe; color: #01579b; border: 1px solid #b3e5fc; }

    /* Custom sidebar header */
    .sidebar-header {
        font-weight: 700;
        font-size: 1.4rem;
        color: #2e7d32;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #c8e6c9;
        padding-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Imports from src subfolder
from src.data_generator import generate_complete_dataset, CROP_PROFILES
from src.model import train_pipeline, load_pipeline, predict_realtime, CROPS_LIST

# Firebase Configuration
firebaseConfig = {
    "apiKey": "AIzaSyC2akXH7pcxFbCsbzERxX0BZvC_j4HU-pI",
    "authDomain": "YOUR_PROJECT.firebaseapp.com",
    "databaseURL": "https://smartagriml-d3497-default-rtdb.asia-southeast1.firebasedatabase.app/",
    "projectId": "smartagriml-d3497",
    "storageBucket": "YOUR_PROJECT.appspot.com",
    "messagingSenderId": "XXXX",
    "appId": "XXXX"
}

firebase = pyrebase.initialize_app(firebaseConfig)
db = firebase.database()

# Ensure directories exist
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "precision_agriculture_data.csv")

# 1. Pipeline Auto-Initialization
pipeline = load_pipeline()
if pipeline is None:
    # Trigger dynamic generation and training overlay
    with st.spinner("🚀 AgriShield AI Engine starting up... Generating 5,000 crop dataset records and training Random Forest classifiers..."):
        # Generate dataset
        generate_complete_dataset(samples_per_crop=1000, output_dir=DATA_DIR)
        # Train ML Pipeline
        train_pipeline(CSV_PATH)
        # Load again
        pipeline = load_pipeline()
    st.success("🤖 Neural-Agri Engine initialized successfully! 3 ML Models are loaded.")

# Initialize Session States
if "sensor_history" not in st.session_state:
    st.session_state["sensor_history"] = []
if "current_sensors" not in st.session_state:
    st.session_state["current_sensors"] = {}
if "last_sensors" not in st.session_state:
    st.session_state["last_sensors"] = {}


# App branding and description header
st.markdown("""
<div class="app-header">
    <h1>AgriShield IoT™</h1>
    <p>Real-Time Smart Agriculture Monitoring & AI Prediction System</p>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROL CENTER ---
with st.sidebar:
    st.markdown('<div class="sidebar-header">🚜 AgriShield IoT Core</div>', unsafe_allow_html=True)
    
    selected_crop = st.selectbox(
        "🌾 Target Regional Crop",
        options=CROPS_LIST,
        index=1  # Default to Ragi
    )
    
    # Display crop profile properties in an expander
    p = CROP_PROFILES[selected_crop]
    with st.expander(f"ℹ️ {selected_crop} Optimal Bounds"):
        st.markdown(f"""
        - **Soil Moisture**: {p['ideal_moisture'][0]}% - {p['ideal_moisture'][1]}%
        - **Soil Temp**: {p['ideal_soil_temp'][0]}°C - {p['ideal_soil_temp'][1]}°C
        - **Ideal pH**: {p['ideal_ph'][0]} - {p['ideal_ph'][1]}
        - **Humidity**: {p['ideal_humidity'][0]}% - {p['ideal_humidity'][1]}%
        """)
        
    st.markdown('<div style="margin-top:1.5rem;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-header">📡 Telemetry Control</div>', unsafe_allow_html=True)
    
    scenario = st.selectbox(
        "🚨 Environmental Preset",
        options=["Normal", "Drought", "Heavy Rain", "Animal Intrusion", "Compost Rotting", "Soil Acidification"],
        index=0
    )
    
    # Trigger tick logic
    tick_btn = st.button("📡 Fetch Next Telemetry Tick", type="primary", use_container_width=True)
    
    # Automatic drift toggle
    auto_refresh_secs = st.slider("🔄 Auto Sync Interval (seconds)", min_value=0, max_value=15, value=0, help="Set to 0 to disable auto-refresh. Requires clicking the button manually.")

# Read Live Sensor Data From Firebase
# ================= FIREBASE LIVE SENSOR DATA =================

firebase_data = db.child("sensor").get()

sensor = firebase_data.val()

if sensor is None:
    st.error("❌ No sensor data found in Firebase.")
    st.stop()

curr = {
    "Soil_Moisture": float(sensor.get("soil_moisture", 0)),
    "Soil_Temp": float(sensor.get("soil_temperature", 0)),
    "pH": float(sensor.get("ph", 0)),
    "MQ135_Gas": float(sensor.get("gas", 0)),

    # You currently don't upload these from ESP32
    "Ambient_Temp": 0,
    "Ambient_Humidity": float(sensor.get("humidity", 0)),
    "SW420_Vibration": 0,

    # Dummy system values for now
    "ESP32_Voltage": 3.3,
    "ESP32_RSSI": -55,
    "ESP32_Battery": 85
}
# Store Firebase live data into history
hist_entry = curr.copy()
hist_entry["Timestamp"] = pd.Timestamp.now()

st.session_state["sensor_history"].append(hist_entry)

# Limit history
if len(st.session_state["sensor_history"]) > 50:
    st.session_state["sensor_history"].pop(0)
deltas = {
    "Soil_Moisture": 0,
    "Soil_Temp": 0,
    "pH": 0,
    "MQ135_Gas": 0,
    "Ambient_Temp": 0,
    "SW420_Vibration": 0
}

# Set up Auto-Refresh using Streamlit timer injection (if requested)
if auto_refresh_secs > 0:
    time.sleep(0.1)  # small buffer
    st.rerun()  # forces the loop drift

# --- TABS CREATION ---
tab1, tab2, tab3 = st.tabs([
    "🚜 Live Farm & AI Advisor", 
    "📊 Historical Trends", 
    "🧠 ML Controller & Workshop"
])

# ================= TAB 1: LIVE MONITORING & ADVISOR =================
with tab1:
    # 1. ESP32 Telemetry Bar
    bat_val = curr["ESP32_Battery"]
    rssi_val = curr["ESP32_RSSI"]
    v_val = curr["ESP32_Voltage"]
    
    wifi_icon = "📶" if rssi_val > -65 else "⚠️📶"
    bat_icon = "🔋" if bat_val > 25 else "🪫"
    
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.7); padding:10px 20px; border-radius:12px; display:flex; justify-content:space-between; margin-bottom:20px; font-weight:600; border:1px solid #d0e8dd;">
        <div>⚡ ESP32 Core Status: <span style="color:#2e7d32;">ONLINE</span></div>
        <div>{wifi_icon} WiFi Signal: <span style="color:#2e7d32;">{rssi_val:.1f} dBm</span></div>
        <div>{bat_icon} Solar Battery: <span style="color:#2e7d32;">{bat_val:.1f}%</span> ({v_val:.2f}V)</div>
        <div>🎯 Scenario Mode: <span style="color:#ef6c00;">{scenario}</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Metric Grid Layout
    st.subheader("📡 Live Sensor Telemetry (Node 01 - Western Zone)")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric(
        label="Soil Moisture (SEN0161)", 
        value=f"{curr['Soil_Moisture']:.1f}%", 
        delta=f"{deltas['Soil_Moisture']:.1f}%" if deltas['Soil_Moisture'] != 0 else None
    )
    col2.metric(
        label="Soil Temp (DS18B20)", 
        value=f"{curr['Soil_Temp']:.1f}°C", 
        delta=f"{deltas['Soil_Temp']:.1f}°C" if deltas['Soil_Temp'] != 0 else None
    )
    col3.metric(
        label="Soil pH (SEN0161)", 
        value=f"{curr['pH']:.2f}", 
        delta=f"{deltas['pH']:.2f}" if deltas['pH'] != 0 else None
    )
    col4.metric(
        label="MQ-135 Gas", 
        value=f"{curr['MQ135_Gas']:.0f} ppm", 
        delta=f"{deltas['MQ135_Gas']:.0f} ppm" if deltas['MQ135_Gas'] != 0 else None
    )
    
    st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)
    
    # 3. AI Inference Grid
    st.subheader("🧠 Neural-Agri Real-Time Predictions")
    
    predictions = predict_realtime(pipeline, selected_crop, curr)
    
    p_col1, p_col2, p_col3 = st.columns(3)
    
    # A. Soil Health Status
    sh = predictions["Soil_Health"]["prediction"]
    sh_conf = predictions["Soil_Health"]["confidence"] * 100
    sh_badge = "excellent" if sh == "Excellent" else ("good" if sh == "Good" else "degraded")
    
    with p_col1:
        st.markdown(f"""
        <div class="glass-card" style="border-top: 6px solid #4caf50; height: 100%;">
            <div style="font-weight:600; color:#555; text-transform:uppercase; font-size:0.85rem; letter-spacing:0.5px;">Target 01: Soil Health Status</div>
            <div style="font-size:1.8rem; font-weight:700; color:#1b5e20; margin-top:8px;">{sh}</div>
            <div class="prediction-badge badge-{sh_badge}">Confidence: {sh_conf:.1f}%</div>
            <div style="margin-top:15px; font-size:0.9rem; color:#666;">
                <strong>Diagnostic:</strong> {"Nutrient structure is highly balanced. Soil microbial activity is ideal." if sh == "Excellent" else ("Mild moisture or pH deviation. Standard conditions." if sh == "Good" else "Warning! Severe chemical imbalance or severe moisture saturation/dryness. Action required.")}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # B. Crop Risk Level
    cr = predictions["Crop_Risk"]["prediction"]
    cr_conf = predictions["Crop_Risk"]["confidence"] * 100
    cr_badge = "low" if cr == "Low" else ("medium" if cr == "Medium" else "high")
    
    with p_col2:
        st.markdown(f"""
        <div class="glass-card" style="border-top: 6px solid #ff9800; height: 100%;">
            <div style="font-weight:600; color:#555; text-transform:uppercase; font-size:0.85rem; letter-spacing:0.5px;">Target 02: Crop Risk Advisor</div>
            <div style="font-size:1.8rem; font-weight:700; color:#e65100; margin-top:8px;">{cr} Risk</div>
            <div class="prediction-badge badge-{cr_badge}">Confidence: {cr_conf:.1f}%</div>
            <div style="margin-top:15px; font-size:0.9rem; color:#666;">
                <strong>Diagnostic:</strong> {"Conditions are healthy. Extremely safe growth zone." if cr == "Low" else ("Moderate hazard. Close inspection or watering adjustment advised." if cr == "Medium" else "Urgent Action Required! Thermal stress, flooding, drought, or wild animal activity detected.")}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # C. Irrigation Recommendation
    ir = predictions["Irrigation_Req"]["prediction"]
    ir_conf = predictions["Irrigation_Req"]["confidence"] * 100
    ir_badge = "irr-none" if ir == "No Irrigation" else ("irr-light" if ir == "Light Irrigation" else "irr-heavy")
    
    with p_col3:
        st.markdown(f"""
        <div class="glass-card" style="border-top: 6px solid #2196f3; height: 100%;">
            <div style="font-weight:600; color:#555; text-transform:uppercase; font-size:0.85rem; letter-spacing:0.5px;">Target 03: Precision Irrigation Advisor</div>
            <div style="font-size:1.8rem; font-weight:700; color:#0d47a1; margin-top:8px;">{ir}</div>
            <div class="prediction-badge badge-{ir_badge}">Confidence: {ir_conf:.1f}%</div>
            <div style="margin-top:15px; font-size:0.9rem; color:#666;">
                <strong>Action:</strong> {"Keep solenoid valves closed. Soil retains adequate moisture." if ir == "No Irrigation" else ("Trigger drip irrigation line for 15 minutes (~2 liters/sq.m)." if ir == "Light Irrigation" else "Alert: Trigger high-flow localized sprinkler for 45 minutes (~6 liters/sq.m).")}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # 4. Live Alert Board
    st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)
    st.subheader("🚨 Real-Time Security & Environmental Alerts")
    
    alert_triggered = False
    
    # Alert conditions based on physical values
    if curr["SW420_Vibration"] > 60.0:
        alert_triggered = True
        st.error(f"🔔 **CRITICAL SECURITY ALERT:** SW-420 High Vibration detected ({curr['SW420_Vibration']:.1f}). Possible Wild Animal Intrusion or Machinery trespass in the {selected_crop} sector! Triggers active on-field acoustic deterrents and sent SMS alert to farmer.")
        
    if curr["Soil_Moisture"] < 12.0:
        alert_triggered = True
        st.warning(f"⚠️ **WATER SCARCITY ALERT:** Critical moisture depletion detected ({curr['Soil_Moisture']:.1f}%). Crop dehydration imminent. Heavy irrigation is highly advised.")
        
    if curr["pH"] < 4.8 or curr["pH"] > 9.5:
        alert_triggered = True
        st.warning(f"⚠️ **SOIL HEALTH WARN:** Critical pH spike detected ({curr['pH']:.2f}). Highly acidic/alkaline soils locked primary nitrogen uptake. Apply regional neutralizing compost buffers.")
        
    if curr["MQ135_Gas"] > 600.0:
        alert_triggered = True
        st.error(f"🔔 **BIO-HAZARD ALERT:** Toxic air gas density spikes ({curr['MQ135_Gas']:.0f} ppm). Indicates composting decay accumulation or field crop waste burning nearby. Inspect site ventilation.")

    if not alert_triggered:
        st.info("🟢 **SYSTEM STATUS NORMAL:** Core nodes reporting high fidelity signals. Soil biochemical compounds are stabilized. No farmer interventions needed.")

# ================= TAB 2: HISTORICAL TRENDS & CHARTING =================
with tab2:
    st.subheader("📊 Dynamic Agronomic Analytics & Correlation Maps")
    
    # Load history DataFrame
    if len(st.session_state["sensor_history"]) < 2:
        st.warning("⏳ Telemetry history requires more data points to plot. Click 'Fetch Next Telemetry Tick' 5-10 times to seed charts!")
    else:
        df_hist = pd.DataFrame(st.session_state["sensor_history"])
        
        hist_col1, hist_col2 = st.columns(2)
        
        with hist_col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.write("##### 📈 Soil Moisture vs Soil Temp Timeline")
            # Line Chart using Plotly
            fig_moist_temp = go.Figure()
            fig_moist_temp.add_trace(go.Scatter(
                x=df_hist["Timestamp"], y=df_hist["Soil_Moisture"],
                mode='lines+markers', name='Moisture (%)',
                line=dict(color='#2196f3', width=3)
            ))
            fig_moist_temp.add_trace(go.Scatter(
                x=df_hist["Timestamp"], y=df_hist["Soil_Temp"],
                mode='lines+markers', name='Soil Temp (°C)',
                line=dict(color='#ff9800', width=3),
                yaxis="y2"
            ))
            fig_moist_temp.update_layout(
                yaxis=dict(title="Moisture (%)"),
                yaxis2=dict(title="Soil Temp (°C)", overlaying="y", side="right"),
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_moist_temp, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with hist_col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.write("##### 🔬 Soil Chemical & Biological Stress Map (pH vs Gas)")
            fig_ph_gas = px.scatter(
                df_hist, x="pH", y="MQ135_Gas",
                color="Soil_Moisture",
                size="SW420_Vibration",
                color_continuous_scale="Viridis",
                labels={"MQ135_Gas": "MQ-135 Gas (ppm)", "pH": "pH Sensor"}
            )
            fig_ph_gas.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_ph_gas, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Display CSV data log
        with st.expander("📂 View Live Active Data Log (Session Cache)"):
            st.dataframe(df_hist.iloc[::-1], use_container_width=True)

# ================= TAB 3: MACHINE LEARNING CONTROLLER =================
with tab3:
    st.subheader("🧠 Neural-Agri Machine Learning Workshop")
    st.write("Explore validation metrics, feature importance, and dynamically retrain the Random Forest model below.")
    
    # 1. Hyperparameter tuning inputs
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("##### ⚡ Live Hyperparameter Fine-Tuning")
    t_col1, t_col2, t_col3 = st.columns(3)
    
    n_est = t_col1.number_input("Random Forest Estimators (Trees)", min_value=10, max_value=250, value=100, step=10)
    m_dep = t_col2.number_input("Maximum Tree Depth", min_value=3, max_value=30, value=12, step=1)
    split_size = t_col3.slider("Evaluation Test Split (%)", 10, 40, 20)
    
    retrain_clicked = st.button("⚡ Train & Calibrate Random Forest Models Live", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if retrain_clicked:
        with st.spinner("🔄 Preprocessing dataset, one-hot encoding categorical variables, and training 3 RF models..."):
            _, new_metrics = train_pipeline(
                CSV_PATH, 
                n_estimators=int(n_est), 
                max_depth=int(m_dep), 
                random_state=42
            )
            # Reload pipeline
            pipeline = load_pipeline()
            st.success("🎉 Models successfully trained, evaluated, and pickled!")
            
    # Show loaded models statistics
    st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)
    st.write("#### 📈 Active Model Performance Metrics")
    
    met = pipeline["metrics"]
    
    acc_col1, acc_col2, acc_col3 = st.columns(3)
    
    acc_col1.metric("Soil Health Model Accuracy", f"{met['Soil_Health']['accuracy'] * 100:.2f}%")
    acc_col2.metric("Crop Risk Model Accuracy", f"{met['Crop_Risk']['accuracy'] * 100:.2f}%")
    acc_col3.metric("Irrigation Advisor Accuracy", f"{met['Irrigation_Req']['accuracy'] * 100:.2f}%")
    
    st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)
    
    # Tabs inside ML for each model's details
    m_details_tab1, m_details_tab2, m_details_tab3 = st.tabs([
        "🌱 Soil Health Model Breakdown",
        "⚠️ Crop Risk Model Breakdown",
        "💧 Irrigation Advisor Breakdown"
    ])
    
    def render_model_details(target_key):
        target_metrics = met[target_key]
        
        # Grid inside
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.write("##### 📊 Confusion Matrix")
            
            cm = np.array(target_metrics["confusion_matrix"])
            labels = target_metrics["labels"]
            
            fig_cm = px.imshow(
                cm,
                x=labels,
                y=labels,
                color_continuous_scale="Greens",
                text_auto=True,
                labels=dict(x="Predicted", y="True Label")
            )
            fig_cm.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_m2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.write("##### 💡 MDI Feature Importance")
            
            fi = target_metrics["feature_importance"]
            
            # Convert to DataFrame
            df_fi = pd.DataFrame({
                "Feature": list(fi.keys()),
                "MDI Importance": list(fi.values())
            }).sort_values(by="MDI Importance", ascending=True)
            
            fig_fi = px.bar(
                df_fi.tail(10),  # Top 10 features
                x="MDI Importance",
                y="Feature",
                orientation='h',
                color="MDI Importance",
                color_continuous_scale="Viridis"
            )
            fig_fi.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_fi, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
    with m_details_tab1:
        render_model_details("Soil_Health")
        
    with m_details_tab2:
        render_model_details("Crop_Risk")
        
    with m_details_tab3:
        render_model_details("Irrigation_Req")

# Footer pitch
st.markdown("""
---
<div style="text-align:center; color:#555; padding:15px; font-weight:600; font-size:0.95rem;">
    AgriShield IoT™ - Built for Smart Farming & Agritech Startup Pitch Competitions 🌾🇮🇳
</div>
""", unsafe_allow_html=True)
