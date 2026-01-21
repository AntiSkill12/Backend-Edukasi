from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, send_from_directory
import os
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import firebase_admin
from firebase_admin import credentials, firestore
import uuid

# ================= INIT APP =================
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

BASE_URL = os.environ.get("BASE_URL", "")

@app.route("/uploads/<path:filename>")
def serve_uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ================= FIREBASE (ENV BASED) =================
private_key = os.environ.get("FIREBASE_PRIVATE_KEY")

if not private_key:
    raise ValueError("FIREBASE_PRIVATE_KEY is not set")

firebase_config = {
    "type": "service_account",
    "project_id": os.environ.get("FIREBASE_PROJECT_ID"),
    "private_key_id": os.environ.get("FIREBASE_PRIVATE_KEY_ID"),
    "private_key": private_key.replace("\\n", "\n"),
    "client_email": os.environ.get("FIREBASE_CLIENT_EMAIL"),
    "client_id": os.environ.get("FIREBASE_CLIENT_ID"),
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": os.environ.get("FIREBASE_CLIENT_CERT_URL"),
}


cred = credentials.Certificate(firebase_config)
firebase_admin.initialize_app(cred)
db = firestore.client()

# ================= YOLO MODELS =================
models = {
    "Mangga": YOLO("models/Mangga.pt"),
    "Pepaya": YOLO("models/Pepaya.pt"),
    "Pisang": YOLO("models/Pisang.pt"),
}

# Label sesuai training
disease_labels = ["rotten", "tidak_berpenyakit", "anthracnose"]

class_colors = {
    "anthracnose": (255, 0, 0),
    "rotten": (0, 0, 255),
    "tidak_berpenyakit": (0, 255, 0),
}

CONFIDENCE_THRESHOLD = 0.4
MIN_BOX_AREA_RATIO = 0.03

# ================= EDUKASI =================
fruit_info = {
    "Mangga": {
        "Vitamin": "Buah Mangga mengandung Vitamin A, C, dan E.",
        "Manfaat": [
            "Menjaga daya tahan tubuh",
            "Membuat kulit sehat",
            "Melancarkan pencernaan",
        ],
    },
    "Pepaya": {
        "Vitamin": "Buah Pepaya kaya Vitamin A, C, dan E.",
        "Manfaat": [
            "Melancarkan pencernaan",
            "Menjaga kesehatan mata",
            "Meningkatkan daya tahan tubuh",
        ],
    },
    "Pisang": {
        "Vitamin": "Buah Pisang mengandung Vitamin B6, C, dan Kalium.",
        "Manfaat": [
            "Sumber energi cepat",
            "Menjaga kesehatan jantung",
            "Membantu pencernaan",
        ],
    },
}

disease_info = {
    "Mangga": {
        "rotten": {
            "Deteksi": "Rotten",
            "Penyebab": "Mangga busuk akibat penyimpanan terlalu lama atau lembab.",
            "Solusi": "Simpan di tempat sejuk dan kering."
        },
        "anthracnose": {
            "Deteksi": "Anthracnose",
            "Penyebab": "Jamur Colletotrichum.",
            "Solusi": "Hindari kelembaban tinggi."
        }
    },
    "Pepaya": {
        "rotten": {
            "Deteksi": "Rotten",
            "Penyebab": "Jamur di kondisi lembab.",
            "Solusi": "Simpan di tempat kering."
        },
        "anthracnose": {
            "Deteksi": "Anthracnose",
            "Penyebab": "Jamur Colletotrichum.",
            "Solusi": "Pisahkan buah terinfeksi."
        }
    },
    "Pisang": {
        "rotten": {
            "Deteksi": "Rotten",
            "Penyebab": "Pisang terlalu matang.",
            "Solusi": "Simpan di tempat sejuk."
        },
        "anthracnose": {
            "Deteksi": "Anthracnose",
            "Penyebab": "Jamur Colletotrichum.",
            "Solusi": "Kurangi kelembaban."
        }
    },
}

# ================= DETECT =================
@app.route("/detect/<fruit>", methods=["POST"])
def detect(fruit):
    if fruit not in models:
        return jsonify({"error": "Model tidak tersedia"}), 400

    if "image" not in request.files:
        return jsonify({"error": "Image required"}), 400

    image = request.files["image"]
    filename = f"{uuid.uuid4().hex}_{image.filename}"
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(image_path)

    model = models[fruit]
    results = model.predict(source=image_path)

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    img_width, img_height = img.size
    img_area = img_width * img_height

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    best = None

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            conf = float(box.conf)
            if conf >= CONFIDENCE_THRESHOLD:
                cls = disease_labels[int(box.cls)]
                x1, y1, x2, y2 = box.xyxy[0]
                area = (x2 - x1) * (y2 - y1)
                if not best or conf > best["conf"]:
                    best = {
                        "cls": cls,
                        "conf": conf,
                        "box": box.xyxy[0],
                        "area": area,
                    }

    response = {
        "fruit": fruit,
        "status": "Bukan Buah",
        "confidence": 0,
        "image_url": f"{BASE_URL}/uploads/{filename}",
        "result": {
            "info": f"Objek pada gambar bukan buah {fruit}"
        }
    }

    if not best:
        return jsonify(response)

    x1, y1, x2, y2 = map(int, best["box"])
    color = class_colors.get(best["cls"], (255, 255, 255))
    draw.rectangle([x1, y1, x2, y2], outline=color, width=6)

    if best["cls"] == "tidak_berpenyakit" and (best["area"] / img_area) >= MIN_BOX_AREA_RATIO:
        response["status"] = "Sehat"
        response["result"] = fruit_info[fruit]
        label = "Sehat"
    else:
        response["status"] = "Sakit"
        response["result"] = disease_info[fruit][best["cls"]]
        label = best["cls"]

    draw.text((x1, y1 - 10), label, fill=color, font=font)

    output = f"detected_{filename}"
    output_path = os.path.join(UPLOAD_FOLDER, output)
    img.save(output_path)

    response["confidence"] = best["conf"]
    response["image_url"] = f"{BASE_URL}/uploads/{output}"

    return jsonify(response)

# ================= ARTICLE CRUD =================
@app.route("/articles", methods=["POST"])
def create_article():
    data = request.json
    ref = db.collection("articles").add({
        "title": data["title"],
        "content": data["content"],
        "createdAt": firestore.SERVER_TIMESTAMP,
    })
    return jsonify({"id": ref[1].id}), 201

@app.route("/articles", methods=["GET"])
def get_articles():
    docs = db.collection("articles").stream()
    return jsonify([{**d.to_dict(), "id": d.id} for d in docs])

# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
