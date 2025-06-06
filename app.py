from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import pickle
import os
import json
import numpy as np

app = Flask(__name__, static_folder="static")

# Load model and label encoder
model = pickle.load(open("models/career_path_model.pkl", "rb"))
le = pickle.load(open("models/label_encoder.pkl", "rb"))
career_config = pickle.load(open("models/career_config.pkl", "rb"))

feature_order = career_config['feature_order'] 

USERS_FILE = "users/users.json"

app.secret_key = "your_secret_key_here"

@app.route("/logout")
def logout():
    session.clear()  # Clear user session
    return redirect(url_for("login"))

@app.route("/static/<path:filename>")
def static_files(filename):
    return app.send_static_file(filename)

@app.route("/")
def home():
    return redirect(url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        data = request.form
        username = data.get("username")
        password = data.get("password")

        os.makedirs("users", exist_ok=True)

        if not os.path.exists(USERS_FILE) or os.path.getsize(USERS_FILE) == 0:
            with open(USERS_FILE, "w") as f:
                json.dump({}, f)

        with open(USERS_FILE, "r") as f:
            try:
                users = json.load(f)
            except json.JSONDecodeError:
                users = {}

        if username in users:
            return "Username already exists", 400

        users[username] = {"password": password}

        with open(USERS_FILE, "w") as f:
            json.dump(users, f)

        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        data = request.form
        username = data.get("username")
        password = data.get("password")

        
        with open(USERS_FILE, "r") as f:
            users = json.load(f)

        if username not in users or users[username]["password"] != password:
            return "Invalid credentials", 401

        session["student_name"] = username  # or however you get the name
        return redirect(url_for("dashboard"))

    return render_template("login.html")

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    student_name = session.get("student_name")  # Get the student's name from session
    prediction = None

    if request.method == "POST":
        form_key_map = {
            "ENGLISH": "english",
            "MATHEMATICS": "math",
            "SOCIAL STUDIES": "social",
            "AGRIC SCIENCE": "agric",
            "PHE": "phe",
            "BASIC TECH": "btech",
            "COMPUTER": "computer",
            "BUSINESS STUDIES": "business",
            "IRS/CRS": "religious_studies",
            "CCA": "cca",
            "YORUBA": "yoruba"
        }

        scores = []
        for subject in feature_order:
            form_key = form_key_map.get(subject)
            try:
                val = float(request.form.get(form_key, 0))
            except (TypeError, ValueError):
                val = 0.0
            scores.append(val)

        input_data = np.array(scores).reshape(1, -1)
        predicted_label_encoded = model.predict(input_data)
        predicted_career = le.inverse_transform(predicted_label_encoded)[0]
        prediction = f"Predicted Career Path: {predicted_career}"

    return render_template("dashboard.html", prediction=prediction, student_name=student_name)


    prediction = None
    if request.method == "POST":
        # Map form inputs to the exact subject names used during training
        input_dict = {
            'ENGLISH': float(request.form.get("english", 0)),
            'MATHEMATICS': float(request.form.get("math", 0)),
            'SOCIAL STUDIES': float(request.form.get("social", 0)),
            'AGRIC SCIENCE': float(request.form.get("agric", 0)),
            'PHE': float(request.form.get("phe", 0)),
            'BASIC TECH': float(request.form.get("btech", 0)),
            'COMPUTER': float(request.form.get("computer", 0)),
            'BUSIESS STUDIES': float(request.form.get("business", 0)),
            'IRS/CRS': float(request.form.get("religious_studies", 0)),
            'CCA': float(request.form.get("cca", 0)),
            'YORUBA': float(request.form.get("yoruba", 0)),
        }

        # This list MUST be in the same order as used during model training
        available_subjects = [
            'MATHEMATICS', 'BASIC TECH', 'COMPUTER', 'AGRIC SCIENCE', 'PHE', 
            'BUSIESS STUDIES', 'SOCIAL STUDIES', 'ENGLISH', 'YORUBA', 'CCA', 'IRS/CRS'
        ]

        # Build the feature vector in the correct order
        input_scores = [input_dict.get(subj, 0) for subj in available_subjects]
        input_data = np.array(input_scores).reshape(1, -1)

        # Predict using the loaded model and label encoder
        predicted_career = le.inverse_transform(model.predict(input_data))[0]
        prediction = f"Predicted Career Path: {predicted_career}"

    return render_template("dashboard.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
