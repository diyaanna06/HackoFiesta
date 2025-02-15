from flask import Flask, request, jsonify, render_template, make_response
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS
from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import google.generativeai as genai 
app = Flask(__name__)
CORS(app)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing. Make sure it's set in the .env file.")
genai.configure(api_key=GEMINI_API_KEY)

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html") 
@app.route("/chatbot/api", methods=["POST"])
def chatbot_api():
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "Message cannot be empty."}), 400
        bot_response = get_gemini_response(user_message)
        return jsonify({"response": bot_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
def get_gemini_response(user_message):
    try:
        model = genai.GenerativeModel("gemini-pro") 
        response = model.generate_content(user_message)
        if hasattr(response, "text"):
            return response.text 
        return "Oops! The chatbot could not generate a response."
    except Exception as e:
        return f"Error: {str(e)}"
model_path = "RFmodel.pkl"
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)
soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()
soil_encoder.classes_ = np.array(["Black", "Clayey", "Loamy", "Red", "Sandy"])
crop_encoder.classes_ = np.array(["Barley", "Cotton", "Ground Nuts", "Maize", "Millets", 
                                  "Oil seeds", "Paddy", "Pulses", "Sugarcane", "Tobacco",
                                  "Wheat", "coffee", "kidneybeans", "orange", "pomegranate",
                                  "rice", "watermelon"])
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        temperature = float(data.get("temperature", 0))
        humidity = float(data.get("humidity", 0))
        moisture = float(data.get("moisture", 0))
        soil_type = data.get("soil", "").strip()
        crop_type = data.get("crop", "").strip()
        nitrogen = float(data.get("nitrogen", 0))
        potassium = float(data.get("potassium", 0))
        phosphorus = float(data.get("phosphorus", 0))  
        if soil_type in soil_encoder.classes_:
            soil_type_encoded = soil_encoder.transform([soil_type])[0]
        else:
            return jsonify({"error": f"Unknown soil type: {soil_type}"}), 400
        if crop_type in crop_encoder.classes_:
            crop_type_encoded = crop_encoder.transform([crop_type])[0]
        else:
            return jsonify({"error": f"Unknown crop type: {crop_type}"}), 400
        feature_names = ["temperature", "humidity", "moisture", "soil_type", "crop_type", "nitrogen", "potassium", "phosphorus"]
        features = pd.DataFrame([[temperature, humidity, moisture, soil_type_encoded, crop_type_encoded, nitrogen, potassium, phosphorus]], columns=feature_names)
        prediction = model.predict(features)
        response = make_response(jsonify({"fertilizer": prediction[0]}))
        response.headers["Content-Type"] = "application/json"
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    app.run(debug=True)