from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load your trained model
with open("solar_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # Extract input values
        month = float(data["month"])
        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
        wind_speed = float(data["wind_speed"])
        pressure = float(data["pressure"])
        cloud_type = float(data["cloud_type"])
        surface_albedo = float(data["surface_albedo"])
        
        hour = 12  # hardcoded
        #month= 1 #hardcoded

        # Create feature array (adjust order according to your model's training)
        features = np.array([[month, temperature, humidity, wind_speed, pressure, cloud_type, surface_albedo, hour]])

        # Prediction
        prediction = model.predict(features)
        result = float(prediction[0])

        return jsonify({"predicted_solar_energy": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os

    # Check if running in a production environment (PORT is usually set by hosting services)
    port = int(os.environ.get("PORT", 5000))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"

    # Debug mode only if running locally
    debug_mode = not os.environ.get("PORT")

    app.run(host=host, port=port, debug=debug_mode)
