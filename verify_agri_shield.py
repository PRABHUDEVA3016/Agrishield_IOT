import os
import sys

# Add current directory to path so python can find src module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_generator import generate_complete_dataset
from src.model import train_pipeline, load_pipeline, predict_realtime

def main():
    print("====================================================")
    print("  Starting AgriShield IoT System Verification Test  ")
    print("====================================================")
    
    # Define directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    
    print("\n[Step 1/4] Generating 5,000 crop dataset samples...")
    csv_path = generate_complete_dataset(samples_per_crop=1000, output_dir=data_dir)
    print(f"[+] Dataset generated successfully at: {csv_path}")
    
    print("\n[Step 2/4] Training Random Forest Models...")
    payload_path, metrics = train_pipeline(csv_path)
    print(f"[+] Model training completed! Models saved at: {payload_path}")
    
    print("\n[Step 3/4] Model Accuracies:")
    for target, value in metrics.items():
        print(f"   |- {target}: {value['accuracy']*100:.2f}% Accuracy")
        
    print("\n[Step 4/4] Testing Real-Time Inference on Mock ESP32 Telemetry...")
    pipeline = load_pipeline()
    if pipeline is None:
        print("[x] Error: Failed to load trained pipeline.")
        sys.exit(1)
        
    # Mock live readings
    mock_telemetry = {
        "Soil_Moisture": 28.5,       # moderately dry
        "Soil_Temp": 29.4,           # normal
        "pH": 6.8,                   # ideal neutral
        "MQ135_Gas": 150.0,          # clean air
        "Ambient_Temp": 31.0,
        "Ambient_Humidity": 55.0,
        "SW420_Vibration": 2.0,      # quiet
        "ESP32_Voltage": 3.48,
        "ESP32_RSSI": -55.0,
        "ESP32_Battery": 92.0
    }
    
    crops_to_test = ["Ragi", "Rice"]
    for crop in crops_to_test:
        print(f"\n--- Testing Crop-Aware Predictions for: {crop} ---")
        preds = predict_realtime(pipeline, crop, mock_telemetry)
        for target, data in preds.items():
            print(f"   |- {target}: {data['prediction']} (Confidence: {data['confidence']*100:.1f}%)")
            
    print("\n====================================================")
    print("  AgriShield IoT End-to-End System Verified!  ")
    print("====================================================")

if __name__ == "__main__":
    main()
