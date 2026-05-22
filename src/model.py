import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# We will define a rigid list of crops to ensure robust one-hot encoding columns
CROPS_LIST = ["Coconut", "Ragi", "Rice", "Sugarcane", "Wheat"]

NUMERICAL_FEATURES = [
    "Soil_Moisture", "Soil_Temp", "pH", "MQ135_Gas", 
    "Ambient_Temp", "Ambient_Humidity", "SW420_Vibration",
    "ESP32_Voltage", "ESP32_RSSI", "ESP32_Battery"
]

TARGETS = ["Soil_Health", "Crop_Risk", "Irrigation_Req"]

def preprocess_data(df, scaler=None, is_training=True):
    """
    Transforms numerical features and one-hot encodes the categorical 'Crop_Type' feature.
    Ensures that identical column order is maintained during training and realtime inference.
    """
    df_copy = df.copy()
    
    # 1. One-hot encode Crop_Type manually to ensure absolute column ordering
    for crop in CROPS_LIST:
        df_copy[f"Crop_Type_{crop}"] = (df_copy["Crop_Type"] == crop).astype(float)
        
    # Drop original Crop_Type
    if "Crop_Type" in df_copy.columns:
        df_copy = df_copy.drop(columns=["Crop_Type"])
        
    # Get feature list: numerical features + one-hot encoded crops
    feature_cols = NUMERICAL_FEATURES + [f"Crop_Type_{crop}" for crop in CROPS_LIST]
    
    X = df_copy[feature_cols].copy()
    
    # 2. Scale numerical features
    if is_training:
        scaler = StandardScaler()
        X[NUMERICAL_FEATURES] = scaler.fit_transform(X[NUMERICAL_FEATURES])
        return X, scaler, feature_cols
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided for inference preprocessing.")
        X[NUMERICAL_FEATURES] = scaler.transform(X[NUMERICAL_FEATURES])
        return X

def train_pipeline(data_path_or_df, n_estimators=100, max_depth=12, random_state=42):
    """
    Runs the entire machine learning training pipeline:
    - Generates or loads the dataset
    - Preprocesses inputs
    - Splits train/test sets
    - Trains 3 Random Forest Classifiers
    - Evaluates accuracy, feature importances, and confusion matrices
    - Saves all models to disk
    """
    if isinstance(data_path_or_df, str):
        df = pd.read_csv(data_path_or_df)
    else:
        df = data_path_or_df
        
    # 1. Preprocess
    X, scaler, feature_cols = preprocess_data(df, is_training=True)
    
    results = {}
    models = {}
    
    # 2. Train and evaluate for each target
    for target in TARGETS:
        y = df[target]
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )
        
        # Train Random Forest Classifier
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        # Predict & Evaluate
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Confusion Matrix
        labels = sorted(list(y.unique()))
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        
        # Feature Importance
        importances = rf.feature_importances_
        feature_importance_dict = dict(zip(feature_cols, importances))
        # Sort feature importance
        sorted_importance = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))
        
        # Store models and metrics
        models[target] = rf
        results[target] = {
            "accuracy": float(accuracy),
            "confusion_matrix": cm.tolist(),
            "labels": labels,
            "feature_importance": sorted_importance
        }
        
    # 3. Save models and preprocessors to disk
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(workspace_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    pipeline_payload = {
        "scaler": scaler,
        "feature_cols": feature_cols,
        "models": models,
        "metrics": results
    }
    
    payload_path = os.path.join(models_dir, "precision_agri_pipeline.joblib")
    joblib.dump(pipeline_payload, payload_path)
    
    return payload_path, results

def load_pipeline():
    """
    Loads the trained pipeline containing scaler, models, and evaluation metrics from disk.
    """
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    payload_path = os.path.join(workspace_dir, "models", "precision_agri_pipeline.joblib")
    if not os.path.exists(payload_path):
        return None
    return joblib.load(payload_path)

def predict_realtime(pipeline, crop_name, sensor_dict):
    """
    Executes real-time crop-aware inference on a single dictionary of sensor values.
    Returns:
        dict: Predictions for Soil Health, Crop Risk, and Irrigation Recommendation with probabilities.
    """
    if pipeline is None:
        raise ValueError("Pipeline has not been initialized or loaded.")
        
    scaler = pipeline["scaler"]
    models = pipeline["models"]
    
    # 1. Structure the input as a single-row DataFrame
    input_row = {
        "Crop_Type": crop_name,
        "Soil_Moisture": float(sensor_dict["Soil_Moisture"]),
        "Soil_Temp": float(sensor_dict["Soil_Temp"]),
        "pH": float(sensor_dict["pH"]),
        "MQ135_Gas": float(sensor_dict["MQ135_Gas"]),
        "Ambient_Temp": float(sensor_dict["Ambient_Temp"]),
        "Ambient_Humidity": float(sensor_dict["Ambient_Humidity"]),
        "SW420_Vibration": float(sensor_dict["SW420_Vibration"]),
        "ESP32_Voltage": float(sensor_dict.get("ESP32_Voltage", 3.3)),
        "ESP32_RSSI": float(sensor_dict.get("ESP32_RSSI", -60.0)),
        "ESP32_Battery": float(sensor_dict.get("ESP32_Battery", 80.0))
    }
    
    df_input = pd.DataFrame([input_row])
    
    # 2. Preprocess input
    X_scaled = preprocess_data(df_input, scaler=scaler, is_training=False)
    
    # 3. Make predictions for each target
    predictions = {}
    for target in TARGETS:
        clf = models[target]
        pred_label = clf.predict(X_scaled)[0]
        pred_prob = clf.predict_proba(X_scaled)[0]
        
        # Map labels to their probabilities
        prob_dict = dict(zip(clf.classes_, pred_prob))
        
        predictions[target] = {
            "prediction": pred_label,
            "confidence": float(np.max(pred_prob)),
            "probabilities": {k: float(v) for k, v in prob_dict.items()}
        }
        
    return predictions

if __name__ == "__main__":
    # Test execution
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(workspace_dir, "data", "precision_agriculture_data.csv")
    if os.path.exists(csv_path):
        payload_path, metrics = train_pipeline(csv_path)
        print("Test training completed. Pipeline saved at:", payload_path)
        print("Model Accuracies:")
        for t, m in metrics.items():
            print(f"- {t}: {m['accuracy'] * 100:.2f}%")
    else:
        print("No CSV found for training. Run src/data_generator.py first.")
