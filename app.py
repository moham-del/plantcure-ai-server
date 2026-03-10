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

DISEASE_INFO = {
    "Tomato___Late_blight":{"disease":"Tomato Late Blight","plant":"Tomato","severity":"High","medicine":"Mancozeb 75% WP Fungicide","fertilizer":"NPK 10-10-10","solution":["Spray Mancozeb every 7 days","Remove infected leaves immediately","Avoid overhead watering","Ensure good airflow"],"buy_link":"https://www.amazon.in/s?k=mancozeb+fungicide"},
    "Tomato___Early_blight":{"disease":"Tomato Early Blight","plant":"Tomato","severity":"Medium","medicine":"Chlorothalonil Fungicide","fertilizer":"NPK 5-10-10","solution":["Apply Chlorothalonil every 10 days","Remove lower infected leaves","Mulch around base of plants","Water early morning"],"buy_link":"https://www.amazon.in/s?k=chlorothalonil+fungicide"},
    "Potato___Late_blight":{"disease":"Potato Late Blight","plant":"Potato","severity":"High","medicine":"Metalaxyl-M + Mancozeb","fertilizer":"NPK 15-15-15","solution":["Apply Metalaxyl-M immediately","Destroy infected plants","Do not harvest in wet weather"],"buy_link":"https://www.amazon.in/s?k=metalaxyl+fungicide"},
    "Potato___Early_blight":{"disease":"Potato Early Blight","plant":"Potato","severity":"Medium","medicine":"Azoxystrobin Fungicide","fertilizer":"NPK 12-12-17","solution":["Spray Azoxystrobin every 14 days","Remove infected leaves","Improve drainage"],"buy_link":"https://www.amazon.in/s?k=azoxystrobin"},
    "Corn_(maize)___Common_rust_":{"disease":"Corn Common Rust","plant":"Corn","severity":"Medium","medicine":"Propiconazole Fungicide","fertilizer":"NPK 20-10-10","solution":["Apply Propiconazole at first sign","Scout fields weekly","Plant resistant varieties"],"buy_link":"https://www.amazon.in/s?k=propiconazole"},
    "Tomato___healthy":{"disease":"Healthy Leaf","plant":"Tomato","severity":"None","medicine":"No medicine needed","fertilizer":"NPK 10-10-10","solution":["Your plant is healthy! 🎉","Continue regular watering","Apply fertilizer monthly"],"buy_link":""},
    "Potato___healthy":{"disease":"Healthy Leaf","plant":"Potato","severity":"None","medicine":"No medicine needed","fertilizer":"NPK 15-15-15","solution":["Your plant is healthy! 🎉","Continue regular care"],"buy_link":""},
}

def get_info(class_name):
    if class_name in DISEASE_INFO:
        return DISEASE_INFO[class_name]
    for key in DISEASE_INFO:
        if key.lower() in class_name.lower():
            return DISEASE_INFO[key]
    parts = class_name.replace('___',' — ').replace('_',' ')
    return {"disease":parts,"plant":parts.split('—')[0].strip(),"severity":"Medium","medicine":"Consult agricultural expert","fertilizer":"NPK 10-10-10","solution":[f"Disease: {parts}","Consult local agricultural office","Remove infected plants"],"buy_link":"https://www.amazon.in/s?k=fungicide"}

@app.route('/')
def home():
    return jsonify({"status":"✅ PlantCure AI Running!","diseases":len(CLASS_NAMES)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error":"No image"}), 400
        img = Image.open(io.BytesIO(request.files['image'].read())).convert('RGB').resize((224,224))
        arr = np.expand_dims(np.array(img)/255.0, 0)
        preds = model.predict(arr, verbose=0)
        idx = np.argmax(preds[0])
        conf = float(preds[0][idx])*100
        if conf < 40:
            return jsonify({"error":"not_plant","message":"Please upload a clear leaf photo"}), 400
        info = get_info(CLASS_NAMES[idx])
        top3 = [{"class":CLASS_NAMES[i].replace('___',' - ').replace('_',' '),"confidence":round(float(preds[0][i])*100,2)} for i in np.argsort(preds[0])[-3:][::-1]]
        return jsonify({"success":True,"confidence":round(conf,2),"class_name":CLASS_NAMES[idx],**info,"top3":top3})
    except Exception as e:
        return jsonify({"error":str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🌱 PlantCure AI → http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port)