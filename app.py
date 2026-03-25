# ══════════════════════════════════════════
# PlantCure AI — Flask Backend Server
# ══════════════════════════════════════════

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import json, io, os, gdown

app = Flask(__name__)
CORS(app)

# ── Download model from Google Drive if not exists ──
MODEL_PATH = 'plantcure_model.h5'
GDRIVE_ID  = '1o9c1iLRAR2v2zXUQDl8I8ghUlDVVPOk_'

if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Google Drive...")
    gdown.download(f'https://drive.google.com/uc?id={GDRIVE_ID}', MODEL_PATH, quiet=False)
    print("✅ Model downloaded!")

print("🔄 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
with open('class_names.json') as f:
    CLASS_NAMES = json.load(f)
print(f"✅ Ready! {len(CLASS_NAMES)} diseases loaded!")

# ══════════════════════════════════════════
# DISEASE DATABASE
# ══════════════════════════════════════════
DISEASE_INFO = {
    "Tomato___Late_blight":{"disease":"Tomato Late Blight","plant":"Tomato","severity":"High","medicine":"Mancozeb 75% WP Fungicide","fertilizer":"NPK 10-10-10","solution":["Spray Mancozeb every 7 days","Remove infected leaves immediately","Avoid overhead watering","Ensure good airflow"],"buy_link":"https://www.amazon.in/s?k=mancozeb+fungicide"},
    "Tomato___Early_blight":{"disease":"Tomato Early Blight","plant":"Tomato","severity":"Medium","medicine":"Chlorothalonil Fungicide","fertilizer":"NPK 5-10-10","solution":["Apply Chlorothalonil every 10 days","Remove lower infected leaves","Mulch around base of plants","Water early morning"],"buy_link":"https://www.amazon.in/s?k=chlorothalonil+fungicide"},
    "Potato___Late_blight":{"disease":"Potato Late Blight","plant":"Potato","severity":"High","medicine":"Metalaxyl-M + Mancozeb","fertilizer":"NPK 15-15-15","solution":["Apply Metalaxyl-M immediately","Destroy infected plants","Do not harvest in wet weather"],"buy_link":"https://www.amazon.in/s?k=metalaxyl+fungicide"},
    "Potato___Early_blight":{"disease":"Potato Early Blight","plant":"Potato","severity":"Medium","medicine":"Azoxystrobin Fungicide","fertilizer":"NPK 12-12-17","solution":["Spray Azoxystrobin every 14 days","Remove infected leaves","Improve drainage"],"buy_link":"https://www.amazon.in/s?k=azoxystrobin"},
    "Corn_(maize)___Common_rust_":{"disease":"Corn Common Rust","plant":"Corn","severity":"Medium","medicine":"Propiconazole Fungicide","fertilizer":"NPK 20-10-10","solution":["Apply Propiconazole at first sign","Scout fields weekly","Plant resistant varieties"],"buy_link":"https://www.amazon.in/s?k=propiconazole"},
    "Grape___Black_rot":{"disease":"Grape Black Rot","plant":"Grape","severity":"High","medicine":"Myclobutanil Fungicide","fertilizer":"Potassium Sulphate","solution":["Apply Myclobutanil before bloom","Remove mummified berries","Prune to improve air circulation"],"buy_link":"https://www.amazon.in/s?k=myclobutanil"},
    "Apple___Apple_scab":{"disease":"Apple Scab","plant":"Apple","severity":"Medium","medicine":"Captan 50% WP Fungicide","fertilizer":"NPK 10-10-10 + Boron","solution":["Apply Captan from bud break","Rake and destroy fallen leaves","Prune trees for air circulation"],"buy_link":"https://www.amazon.in/s?k=captan+fungicide"},
    "Tomato___healthy":{"disease":"Healthy Leaf ✅","plant":"Tomato","severity":"None","medicine":"No medicine needed","fertilizer":"NPK 10-10-10","solution":["Your plant is perfectly healthy! 🎉","Continue regular watering","Apply fertilizer monthly","Monitor weekly for disease signs"],"buy_link":""},
    "Potato___healthy":{"disease":"Healthy Leaf ✅","plant":"Potato","severity":"None","medicine":"No medicine needed","fertilizer":"NPK 15-15-15","solution":["Your plant is perfectly healthy! 🎉","Continue regular care","Hill up soil regularly"],"buy_link":""},
}

def get_info(class_name):
    if class_name in DISEASE_INFO:
        return DISEASE_INFO[class_name]
    for key in DISEASE_INFO:
        if key.lower() in class_name.lower():
            return DISEASE_INFO[key]
    parts = class_name.replace('___',' — ').replace('_',' ')
    return {
        "disease": parts,
        "plant": parts.split('—')[0].strip(),
        "severity": "Medium",
        "medicine": "Consult local agricultural expert",
        "fertilizer": "NPK 10-10-10 general fertilizer",
        "solution": [f"Disease detected: {parts}", "Consult your local agricultural office", "Remove and isolate infected plants", "Apply general fungicide as precaution"],
        "buy_link": "https://www.amazon.in/s?k=fungicide"
    }

def is_likely_leaf(img_array):
    """Super strict check — only green plant leaves allowed"""
    img = img_array[0]
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]

    total_pixels = img.shape[0] * img.shape[1]

    # ── Check 1: Green dominance ──
    # Leaf pixels must have green > red AND green > blue
    green_pixels = np.sum((g > r * 1.08) & (g > b * 1.08))
    green_ratio = green_pixels / total_pixels
    if green_ratio < 0.20:  # Need at least 20% green pixels
        return False

    # ── Check 2: Average green channel ──
    avg_green = np.mean(g)
    if avg_green < 0.18:  # Too little green overall
        return False

    # ── Check 3: Brightness check ──
    avg_brightness = np.mean(img)
    if avg_brightness > 0.80:  # Too bright = white/gray background
        return False
    if avg_brightness < 0.06:  # Too dark = black object
        return False

    # ── Check 4: Color variance (leaves have texture) ──
    variance = np.var(img)
    if variance < 0.001:  # Solid color = not a leaf
        return False

    # ── Check 5: Green saturation ──
    # True leaves: green channel significantly higher than others
    green_dominance = np.mean(g) - (np.mean(r) + np.mean(b)) / 2
    if green_dominance < 0.03:
        return False

    return True

# ══════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════

@app.route('/')
def home():
    return jsonify({
        "status": "✅ PlantCure AI Running!",
        "diseases": len(CLASS_NAMES),
        "model": "MobileNetV2"
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Empty file"}), 400

        # Preprocess image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        # ── Check if it looks like a leaf ──
        if not is_likely_leaf(img_batch):
            return jsonify({
                "error": "not_plant",
                "message": "இது plant leaf இல்லை! தயவுசெய்து plant leaf photo மட்டும் upload செய்யுங்கள்."
            }), 400

        # Run AI prediction
        predictions = model.predict(img_batch, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx]) * 100

        # ── Strict confidence check ──
        if confidence < 65:
            return jsonify({
                "error": "not_plant",
                "message": "இது plant leaf இல்லை! Only plant leaf photos accepted."
            }), 400

        class_name = CLASS_NAMES[predicted_idx]
        info = get_info(class_name)

        # Top 3 predictions
        top3_idx = np.argsort(predictions[0])[-3:][::-1]
        top3 = [{
            "class": CLASS_NAMES[i].replace('___', ' - ').replace('_', ' '),
            "confidence": round(float(predictions[0][i]) * 100, 2)
        } for i in top3_idx]

        return jsonify({
            "success": True,
            "class_name": class_name,
            "confidence": round(confidence, 2),
            **info,
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
    port = int(os.environ.get('PORT', 5000))
    print(f"\n{'='*50}")
    print(f"🌱 PlantCure AI Server Starting...")
    print(f"🌐 URL     : http://0.0.0.0:{port}")
    print(f"🦠 Diseases: {len(CLASS_NAMES)}")
    print(f"{'='*50}\n")
    app.run(host='0.0.0.0', port=port)