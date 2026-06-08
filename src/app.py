import gradio as gr
import hopsworks
import joblib
import requests
import pandas as pd
import os
import shap
import matplotlib.pyplot as plt
from processor import standardize_data

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
def load_assets():
    """Logs into Hopsworks, downloads the latest model, and initializes SHAP."""
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    
    # Downloading Version 3 (The XGBoost Model)
    model_meta = mr.get_model("aqi_predictor_xgboost", version=3)
    model_dir = model_meta.download()
    model = joblib.load(os.path.join(model_dir, "model.pkl"))
    
    # SHAP Explainer
    explainer = shap.TreeExplainer(model)
    return model, explainer, fs

model, explainer, fs = load_assets()

# ==========================================
# 2. LOGIC: AQI CATEGORIES (Traffic Light)
# ==========================================
def get_aqi_styling(score):
    """Maps numerical AQI prediction to visual categories."""
    score = int(round(score))
    score = max(1, min(score, 5)) # Clamping 1-5
    
    mapping = {
        1: ("Good", "#27ae60", "Safe air, no health risks."),
        2: ("Fair", "#f1c40f", "Acceptable air, minor concerns."),
        3: ("Moderate", "#f39c12", "Sensitive groups: limit outdoor exertion."),
        4: ("Poor", "#e67e22", "Health impacts likely for everyone."),
        5: ("Very Poor", "#c0392b", "Hazardous! Avoid all outdoor activities.")
    }
    return mapping[score]

# ==========================================
# 3. HELPER: DATA ENRICHMENT
# ==========================================
def get_enriched_data():
    """Fetches live API data and merges with 3-day history from Feature Store."""
    lat, lon = 31.5204, 74.3587
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
    response = requests.get(url).json()
    
    data = response['list'][0]['components']
    data['aqi'] = response['list'][0]['main']['aqi']
    data['timestamp'] = pd.Timestamp.now()
    
    # Fetching Lag features from Version 2 Feature Group
    try:
        fg = fs.get_feature_group("aqi_data_lahore", version=2)
        history = fg.read().sort_values('timestamp', ascending=False).head(3)
        data['aqi_t1'] = history.iloc[0]['aqi'] if len(history) > 0 else data['aqi']
        data['aqi_t2'] = history.iloc[1]['aqi'] if len(history) > 1 else data['aqi']
        data['aqi_t3'] = history.iloc[2]['aqi'] if len(history) > 2 else data['aqi']
    except:
        data['aqi_t1'] = data['aqi_t2'] = data['aqi_t3'] = data['aqi']

    df = pd.DataFrame([data])
    return standardize_data(df, is_training=False)

# ==========================================
# 4. PREDICTION & EXPLANATION
# ==========================================
def get_prediction():
    processed_df = get_enriched_data()
    features = processed_df.drop(['aqi', 'timestamp'], axis=1, errors='ignore').copy()
    
    predictions = []
    current_features = features.copy()
    
    # Recursive Forecasting Loop
    for i in range(3):
        pred = model.predict(current_features)[0]
        predictions.append(pred)
        # Shift lags
        current_features['aqi_t3'] = current_features['aqi_t2']
        current_features['aqi_t2'] = current_features['aqi_t1']
        current_features['aqi_t1'] = pred
    
    # Generate HTML Cards
    html_output = "<div style='display: flex; flex-direction: column; gap: 12px; font-family: sans-serif;'>"
    for i, p in enumerate(predictions):
        label, color, desc = get_aqi_styling(p)
        html_output += f"""
        <div style='background-color: {color}; padding: 16px; border-radius: 12px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h2 style='margin: 0;'>Day {i+1}: {label}</h2>
            <p style='margin: 5px 0 0 0; font-size: 1.1em;'>{desc}</p>
            <small>Predicted Index: {p:.2f}</small>
        </div>
        """
    html_output += "</div>"
    return html_output

def get_shap_plot():
    processed_df = get_enriched_data()
    features = processed_df.drop(['aqi', 'timestamp'], axis=1, errors='ignore')
    
    plt.figure(figsize=(10, 5))
    shap_values = explainer(features)
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig("shap_plot.png")
    plt.close()
    return "shap_plot.png"

# ==========================================
# 5. GRADIO INTERFACE
# ==========================================
with gr.Blocks(theme=gr.themes.Soft(primary_hue="emerald")) as demo:
    gr.Markdown("# 🌬️ Pearls AQI Predictor")
    gr.Markdown("### Advanced Machine Learning for Lahore's Air Quality")
    
    with gr.Row():
        predict_btn = gr.Button("🚀 Generate Forecast", variant="primary")
        shap_btn = gr.Button("📊 Explain Prediction (SHAP)")
        
    with gr.Row():
        output_html = gr.HTML(label="Forecast")
        output_image = gr.Image(label="Feature Importance")
    
    predict_btn.click(fn=get_prediction, outputs=output_html)
    shap_btn.click(fn=get_shap_plot, outputs=output_image)

demo.launch(share=True)