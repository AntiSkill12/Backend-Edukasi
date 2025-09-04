from flask import Flask, request, jsonify
import os
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import firebase_admin
from firebase_admin import credentials, firestore

# === Init Flask ===
app = Flask(__name__)

# === Load Firebase ===
cred = credentials.Certificate("./app/edukasi-penyakit-buah-firebase-adminsdk-fbsvc-0310db69b4.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# === Load 3 Model ===
models = {
    "Mangga": YOLO("models/Mangga.pt"),
    "Pepaya": YOLO("models/Pepaya.pt"),
    "Pisang": YOLO("models/Pisang.pt"),
}

# Label sesuai urutan training model
disease_labels = ["Rotten", "Tidak_Berpenyakit", "anthracnose"]

class_colors = {
    "anthracnose": (255, 0, 0),
    "Rotten": (0, 0, 255),
    "Tidak_Berpenyakit": (0, 255, 0)
}

# === Edukasi jika buah sehat ===
fruit_info = {
    "Mangga": {
        "Vitamin": "Buah Mangga mengandung Vitamin A, C, dan E.",
        "Manfaat": [
            "Menjaga Daya Tahan Tubuh: Vitamin C-nya membantu menjaga tubuh dari serangan penyakit.",
            "Membuat Kulit Sehat dan Bersinar: Vitamin A dan E-nya membantu kulit tetap lembut dan bercahaya.",
            "Melancarkan Pencernaan: Seratnya bertindak seperti sapu kecil di dalam perut."
        ]
    },
    "Pepaya": {
        "Vitamin": "Buah Pepaya kaya akan Vitamin A, C, dan E.",
        "Manfaat": [
            "Melancarkan Pencernaan: Pepaya punya enzim papain yang membantu perut.",
            "Menjaga Kesehatan Mata: Kaya Vitamin A yang membantu penglihatan.",
            "Meningkatkan Daya Tahan Tubuh: Vitamin C membuat tubuh jarang sakit."
        ]
    },
    "Pisang": {
        "Vitamin": "Buah Pisang mengandung Vitamin B6, C, dan Kalium.",
        "Manfaat": [
            "Sumber Energi Cepat: Gula alaminya langsung diubah menjadi tenaga.",
            "Menjaga Kesehatan Jantung: Kalium menjaga detak jantung tetap stabil.",
            "Membantu Pencernaan: Serat membantu makanan dicerna dengan baik."
        ]
    }
}

# === Edukasi penyakit per buah ===
disease_info = {
    "Mangga": {
        "Rotten": {
            "Deteksi": "Rotten",
            "Penyebab": "Mangga bisa busuk karena kelembaban tinggi atau penyimpanan terlalu lama.",
            "Solusi": "Simpan di tempat sejuk, hindari luka pada kulit, konsumsi segera setelah matang."
        },
        "anthracnose": {
            "Deteksi": "Anthracnose",
            "Penyebab": "Disebabkan jamur Colletotrichum, muncul bercak hitam cekung pada kulit.",
            "Solusi": "Buang bagian terinfeksi, simpan di tempat kering, hindari kelembaban tinggi."
        }
    },
    "Pepaya": {
        "Rotten": {
            "Deteksi": "Rotten",
            "Penyebab": "Pepaya busuk karena jamur Phytophthora di kondisi lembab.",
            "Solusi": "Simpan di tempat kering, pisahkan buah rusak dari yang sehat."
        },
        "anthracnose": {
            "Deteksi": "Anthracnose",
            "Penyebab": "Jamur Colletotrichum, bercak hitam berair pada kulit buah.",
            "Solusi": "Buang pepaya terinfeksi, kurangi kelembaban, jaga kebersihan tempat simpan."
        }
    },
    "Pisang": {
        "Rotten": {
            "Deteksi": "Rotten",
            "Penyebab": "Pisang busuk oleh jamur Fusarium atau Colletotrichum saat terlalu matang.",
            "Solusi": "Simpan pisang di tempat sejuk dan kering, jangan menumpuk terlalu lama."
        },
        "anthracnose": {
            "Deteksi": "Anthracnose",
            "Penyebab": "Jamur Colletotrichum, bercak hitam kecil meluas di kulit pisang.",
            "Solusi": "Hindari kelembaban tinggi, pisahkan pisang terinfeksi dari yang sehat."
        }
    }
}

# === Threshold untuk confidence minimal ===
CONFIDENCE_THRESHOLD = 0.3


# ====================== DETECT ENDPOINT ======================
@app.route("/detect/<fruit>", methods=["POST"])
def detect(fruit):
    if fruit not in models:
        return jsonify({"error": f"Model untuk {fruit} tidak ada"}), 400

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    # === Ambil user_id dari request (form-data atau JSON) ===
    user_id = None
    if "user_id" in request.form:
        user_id = request.form["user_id"]
    elif request.is_json and "user_id" in request.json:
        user_id = request.json["user_id"]

    if not user_id:
        return jsonify({"error": "user_id is required"}), 400


    # === Simpan gambar sementara ===
    image = request.files["image"]
    os.makedirs("uploads", exist_ok=True)
    image_path = os.path.join("uploads", image.filename)
    image.save(image_path)

    # === Prediksi pakai YOLO ===
    model = models[fruit]
    results = model.predict(source=image_path)

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    best_detection = None
    for result in results:
        for box in result.boxes:
            confidence = float(box.conf)
            if confidence >= CONFIDENCE_THRESHOLD:
                class_index = int(box.cls)
                disease_name = disease_labels[class_index]

                if (best_detection is None) or (confidence > best_detection["confidence"]):
                    best_detection = {
                        "disease_name": disease_name,
                        "confidence": confidence,
                        "box": box.xyxy[0]
                    }

    # === Default response ===
    response_data = {
        "fruit": fruit,
        "status": "Tidak Terdeteksi",
        "confidence": 0,
        "image_url": "",
        "result": {},
        "user_id": user_id
    }

    if best_detection:
        x_min, y_min, x_max, y_max = best_detection["box"]
        color = class_colors.get(best_detection["disease_name"], (255, 255, 255))
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)

        if best_detection["disease_name"] == "Tidak_Berpenyakit":
            response_data.update({
                "status": "Sehat",
                "confidence": best_detection["confidence"],
                "result": {
                    "Vitamin": fruit_info[fruit]["Vitamin"],
                    "Manfaat": fruit_info[fruit]["Manfaat"]
                }
            })
            label_text = f"Sehat {best_detection['confidence']:.2f}"
        else:
            response_data.update({
                "status": "Sakit",
                "confidence": best_detection["confidence"],
                "result": disease_info[fruit][best_detection["disease_name"]]
            })
            label_text = f"{best_detection['disease_name']} {best_detection['confidence']:.2f}"

        draw.text((x_min, y_min - 10), label_text, font=font, fill=color)

    # === Simpan hasil deteksi ke file ===
    output_path = os.path.join("uploads", f"detected_{fruit}_{image.filename}")
    img.save(output_path)
    response_data["image_url"] = output_path

    # === Simpan ke Firestore dengan user_id ===
    db.collection("detections").add({
        "user_id": user_id,
        "fruit": response_data["fruit"],
        "status": response_data["status"],
        "confidence": response_data["confidence"],
        "result": response_data["result"],
        "image_url": response_data["image_url"],
        "created_at": firestore.SERVER_TIMESTAMP
    })

    return jsonify(response_data)

# ====================== DETECTIONS CRUD ======================

# Get semua detection berdasarkan user_id
@app.route("/detect/<user_id>", methods=["GET"])
def get_detections_by_user(user_id):
    detections_ref = db.collection("detections").where("user_id", "==", user_id).stream()
    detections = []
    for doc in detections_ref:
        det = doc.to_dict()
        det["id"] = doc.id
        detections.append(det)

    if not detections:
        return jsonify({"error": "No detections found for this user"}), 404

    return jsonify(detections), 200


# Get detection berdasarkan user_id + detection_id
@app.route("/detect/<user_id>/<detection_id>", methods=["GET"])
def get_detection_by_user_and_id(user_id, detection_id):
    doc = db.collection("detections").document(detection_id).get()
    if doc.exists:
        det = doc.to_dict()
        if det["user_id"] != user_id:
            return jsonify({"error": "Detection not found for this user"}), 404
        det["id"] = doc.id
        return jsonify(det), 200
    return jsonify({"error": "Detection not found"}), 404


# Delete detection berdasarkan user_id + detection_id
@app.route("/detect/<user_id>/<detection_id>", methods=["DELETE"])
def delete_detection_by_user_and_id(user_id, detection_id):
    doc_ref = db.collection("detections").document(detection_id)
    doc = doc_ref.get()
    if doc.exists:
        det = doc.to_dict()
        if det["user_id"] != user_id:
            return jsonify({"error": "Detection not found for this user"}), 404
        doc_ref.delete()
        return jsonify({"message": f"Detection {detection_id} deleted for user {user_id}"}), 200
    return jsonify({"error": "Detection not found"}), 404




# ====================== USER CRUD ======================
@app.route("/users", methods=["POST"])
def create_user():
    data = request.json
    if not data or "email" not in data or "password" not in data or "name" not in data:
        return jsonify({"error": "Email, Password, and Name are required"}), 400

    doc_ref = db.collection("users").add({
        "email": data["email"],
        "password": data["password"],  # NOTE: sebaiknya di-hash (misal pakai bcrypt)
        "name": data["name"],
        "createdAt": firestore.SERVER_TIMESTAMP
    })

    return jsonify({"message": "User created", "id": doc_ref[1].id}), 201


@app.route("/users", methods=["GET"])
def get_users():
    users_ref = db.collection("users").stream()
    users = []
    for doc in users_ref:
        user = doc.to_dict()
        user["id"] = doc.id
        users.append(user)
    return jsonify(users), 200


@app.route("/users/<user_id>", methods=["GET"])
def get_user(user_id):
    doc = db.collection("users").document(user_id).get()
    if doc.exists:
        user = doc.to_dict()
        user["id"] = doc.id
        return jsonify(user), 200
    return jsonify({"error": "User not found"}), 404


@app.route("/users/<user_id>", methods=["PUT"])
def update_user(user_id):
    data = request.json
    doc_ref = db.collection("users").document(user_id)
    if not doc_ref.get().exists:
        return jsonify({"error": "User not found"}), 404

    update_data = {}
    if "email" in data:
        update_data["email"] = data["email"]
    if "password" in data:
        update_data["password"] = data["password"]
    if "name" in data:
        update_data["name"] = data["name"]

    if update_data:
        doc_ref.update(update_data)

    return jsonify({"message": "User updated"}), 200


@app.route("/users/<user_id>", methods=["DELETE"])
def delete_user(user_id):
    doc_ref = db.collection("users").document(user_id)
    if not doc_ref.get().exists:
        return jsonify({"error": "User not found"}), 404

    doc_ref.delete()
    return jsonify({"message": "User deleted"}), 200


# ====================== LOGIN ======================
@app.route("/login", methods=["POST"])
def login_user():
    data = request.json
    if not data or "email" not in data or "password" not in data:
        return jsonify({"error": "Email and Password are required"}), 400

    users_ref = db.collection("users").where("email", "==", data["email"]).where("password", "==", data["password"]).stream()

    user_data = None
    for user in users_ref:
        user_data = user.to_dict()
        user_data["id"] = user.id

    if user_data:
        return jsonify({"message": "Login success", "user": user_data}), 200
    else:
        return jsonify({"error": "Invalid email or password"}), 401




# ====================== ARTICLE CRUD ======================
@app.route("/articles", methods=["POST"])
def create_article():
    data = request.json
    if not data or "title" not in data or "content" not in data:
        return jsonify({"error": "Title and Content are required"}), 400

    doc_ref = db.collection("articles").add({
        "title": data["title"],
        "content": data["content"],
        "createdAt": firestore.SERVER_TIMESTAMP
    })

    return jsonify({"message": "Article created", "id": doc_ref[1].id}), 201


@app.route("/articles", methods=["GET"])
def get_articles():
    articles_ref = db.collection("articles").stream()
    articles = []
    for doc in articles_ref:
        article = doc.to_dict()
        article["id"] = doc.id
        articles.append(article)
    return jsonify(articles), 200


@app.route("/articles/<article_id>", methods=["GET"])
def get_article(article_id):
    doc = db.collection("articles").document(article_id).get()
    if doc.exists:
        article = doc.to_dict()
        article["id"] = doc.id
        return jsonify(article), 200
    return jsonify({"error": "Article not found"}), 404


@app.route("/articles/<article_id>", methods=["PUT"])
def update_article(article_id):
    data = request.json
    doc_ref = db.collection("articles").document(article_id)
    if not doc_ref.get().exists:
        return jsonify({"error": "Article not found"}), 404

    update_data = {}
    if "title" in data:
        update_data["title"] = data["title"]
    if "content" in data:
        update_data["content"] = data["content"]

    if update_data:
        doc_ref.update(update_data)

    return jsonify({"message": "Article updated"}), 200


@app.route("/articles/<article_id>", methods=["DELETE"])
def delete_article(article_id):
    doc_ref = db.collection("articles").document(article_id)
    if not doc_ref.get().exists:
        return jsonify({"error": "Article not found"}), 404

    doc_ref.delete()
    return jsonify({"message": "Article deleted"}), 200


# ====================== MAIN ======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
