from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import io
import base64
import os

app = Flask(__name__)
CORS(app)  # Allow website to connect

# ── Load AI Model ──
print("🔄 Loading PlantCure AI model...")
model = tf.keras.models.load_model('plantcure_model.h5')

# ── Load class names ──
with open('class_names.json', 'r') as f:
    CLASS_NAMES = json.load(f)

print(f"✅ Model loaded! {len(CLASS_NAMES)} diseases ready!")

# ══════════════════════════════════════════
# DISEASE SOLUTIONS DATABASE
# ══════════════════════════════════════════
DISEASE_INFO = {
    "Tomato___Late_blight": {
        "disease": "Tomato Late Blight",
        "plant": "Tomato",
        "severity": "High",
        "medicine": "Mancozeb 75% WP Fungicide",
        "fertilizer": "NPK 10-10-10 + Calcium Nitrate",
        "solution": [
            "Spray Mancozeb (2.5g/L water) every 7 days",
            "Remove all infected leaves immediately and burn them",
            "Avoid overhead watering — water at soil level only",
            "Ensure good airflow between plants",
            "Apply NPK fertilizer after rain"
        ],
        "buy_link": "https://www.amazon.in/s?k=mancozeb+fungicide"
    },
    "Tomato___Early_blight": {
        "disease": "Tomato Early Blight",
        "plant": "Tomato",
        "severity": "Medium",
        "medicine": "Chlorothalonil Fungicide",
        "fertilizer": "NPK 5-10-10",
        "solution": [
            "Apply Chlorothalonil spray every 10 days",
            "Remove lower infected leaves carefully",
            "Mulch around base of plants to prevent soil splash",
            "Water early morning so leaves dry quickly",
            "Rotate crops next season"
        ],
        "buy_link": "https://www.amazon.in/s?k=chlorothalonil+fungicide"
    },
    "Potato___Late_blight": {
        "disease": "Potato Late Blight",
        "plant": "Potato",
        "severity": "High",
        "medicine": "Metalaxyl-M + Mancozeb",
        "fertilizer": "NPK 15-15-15",
        "solution": [
            "Apply Metalaxyl-M fungicide immediately",
            "Hill up soil around plants to protect tubers",
            "Destroy all infected plant material by burning",
            "Do not harvest during wet weather",
            "Store harvested potatoes in cool dry place"
        ],
        "buy_link": "https://www.amazon.in/s?k=metalaxyl+fungicide"
    },
    "Potato___Early_blight": {
        "disease": "Potato Early Blight",
        "plant": "Potato",
        "severity": "Medium",
        "medicine": "Azoxystrobin Fungicide",
        "fertilizer": "NPK 12-12-17",
        "solution": [
            "Spray Azoxystrobin every 14 days",
            "Remove and destroy infected leaves",
            "Improve field drainage to reduce moisture",
            "Apply balanced fertilizer to boost plant immunity",
            "Plant resistant varieties next season"
        ],
        "buy_link": "https://www.amazon.in/s?k=azoxystrobin+fungicide"
    },
    "Corn_(maize)___Common_rust_": {
        "disease": "Corn Common Rust",
        "plant": "Corn / Maize",
        "severity": "Medium",
        "medicine": "Propiconazole Fungicide",
        "fertilizer": "NPK 20-10-10 + Zinc",
        "solution": [
            "Apply Propiconazole spray at first sign of rust",
            "Scout fields weekly during humid conditions",
            "Plant rust-resistant hybrid varieties",
            "Apply zinc micronutrient for better immunity",
            "Ensure proper plant spacing for air circulation"
        ],
        "buy_link": "https://www.amazon.in/s?k=propiconazole+fungicide"
    },
    "Grape___Black_rot": {
        "disease": "Grape Black Rot",
        "plant": "Grape",
        "severity": "High",
        "medicine": "Myclobutanil Fungicide",
        "fertilizer": "Potassium Sulphate + Calcium",
        "solution": [
            "Apply Myclobutanil before and after bloom",
            "Remove all mummified berries from vines",
            "Prune to improve air circulation in canopy",
            "Clean up fallen leaves and debris",
            "Apply protective spray before rainy periods"
        ],
        "buy_link": "https://www.amazon.in/s?k=myclobutanil+fungicide"
    },
    "Apple___Apple_scab": {
        "disease": "Apple Scab",
        "plant": "Apple",
        "severity": "Medium",
        "medicine": "Captan 50% WP Fungicide",
        "fertilizer": "NPK 10-10-10 + Boron",
        "solution": [
            "Apply Captan fungicide from bud break",
            "Rake and destroy fallen infected leaves",
            "Prune trees to improve air circulation",
            "Apply during wet spring weather every 7-10 days",
            "Plant scab-resistant apple varieties"
        ],
        "buy_link": "https://www.amazon.in/s?k=captan+fungicide"
    },
    "Tomato___healthy": {
        "disease": "Healthy Leaf",
        "plant": "Tomato",
        "severity": "None",
        "medicine": "No medicine needed",
        "fertilizer": "Continue NPK 10-10-10 regularly",
        "solution": [
            "Your plant is perfectly healthy! 🎉",
            "Continue regular watering schedule",
            "Apply balanced NPK fertilizer monthly",
            "Monitor weekly for any early disease signs",
            "Maintain good air circulation around plants"
        ],
        "buy_link": ""
    },
    "Potato___healthy": {
        "disease": "Healthy Leaf",
        "plant": "Potato",
        "severity": "None",
        "medicine": "No medicine needed",
        "fertilizer": "Continue NPK 15-15-15 regularly",
        "solution": [
            "Your plant is perfectly healthy! 🎉",
            "Continue regular watering schedule",
            "Hill up soil around plants regularly",
            "Monitor weekly for any disease signs",
            "Apply potassium rich fertilizer for tuber growth"
        ],
        "buy_link": ""
    }
}

def get_disease_info(class_name):
    """Get disease info — check exact match first, then partial"""
    # Exact match
    if class_name in DISEASE_INFO:
        return DISEASE_INFO[class_name]

    # Partial match
    for key in DISEASE_INFO:
        if key.lower() in class_name.lower() or class_name.lower() in key.lower():
            return DISEASE_INFO[key]

    # Default response
    parts = class_name.replace('___', ' — ').replace('_', ' ')
    return {
        "disease": parts,
        "plant": parts.split('—')[0].strip() if '—' in parts else "Unknown Plant",
        "severity": "Medium",
        "medicine": "Consult local agricultural expert",
        "fertilizer": "NPK 10-10-10 general fertilizer",
        "solution": [
            f"Disease detected: {parts}",
            "Consult your local agricultural office",
            "Remove and isolate infected plants",
            "Apply general fungicide as precaution",
            "Monitor other plants for similar symptoms"
        ],
        "buy_link": "https://www.amazon.in/s?k=plant+fungicide"
    }

# ══════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════

@app.route('/')
def home():
    return jsonify({
        "status": "✅ PlantCure AI Server is Running!",
        "version": "1.0.0",
        "diseases": len(CLASS_NAMES),
        "model": "MobileNetV2 + Custom Layers"
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ── Get image from request ──
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Empty file"}), 400

        # ── Preprocess image ──
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ── Run AI prediction ──
        predictions = model.predict(img_array, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx]) * 100

        class_name = CLASS_NAMES[predicted_idx]

        # ── Check if it's a plant leaf ──
        if confidence < 40:
            return jsonify({
                "error": "not_plant",
                "message": "Please upload a clear plant leaf image"
            }), 400

        # ── Get disease info ──
        info = get_disease_info(class_name)

        # ── Get top 3 predictions ──
        top3_idx = np.argsort(predictions[0])[-3:][::-1]
        top3 = []
        for idx in top3_idx:
            top3.append({
                "class": CLASS_NAMES[idx].replace('___', ' - ').replace('_', ' '),
                "confidence": round(float(predictions[0][idx]) * 100, 2)
            })

        return jsonify({
            "success": True,
            "class_name": class_name,
            "confidence": round(confidence, 2),
            "disease": info["disease"],
            "plant": info["plant"],
            "severity": info["severity"],
            "medicine": info["medicine"],
            "fertilizer": info["fertilizer"],
            "solution": info["solution"],
            "buy_link": info["buy_link"],
            "top3": top3
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model_loaded": True})

# ══════════════════════════════════════════
# START SERVER
# ══════════════════════════════════════════
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🌱 PlantCure AI Server Starting...")
    print("="*50)
    print(f"🌐 Website : http://localhost:5000")
    print(f"🤖 AI Model: plantcure_model.h5")
    print(f"🦠 Diseases: {len(CLASS_NAMES)}")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)