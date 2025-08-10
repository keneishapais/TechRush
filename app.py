
from flask import Flask, request, jsonify
import requests
import joblib
import os
from datetime import datetime
from flask_cors import CORS

# Load your ML model
MODEL_PATH = "solar_model.pkl"
model = joblib.load(MODEL_PATH)


# API keys
OPENCAGE_API_KEY = "OPEN_CAGE_API_KEY"

# Flask setup
app = Flask(__name__)
CORS(app)

# Cloud % → model category mapping
def map_cloud_percent_to_type(cloud_percent):
    if cloud_percent < 5: return 0
    elif cloud_percent < 20: return 1
    elif cloud_percent < 35: return 2
    elif cloud_percent < 50: return 3
    elif cloud_percent < 65: return 4
    elif cloud_percent < 75: return 5
    elif cloud_percent < 85: return 6
    elif cloud_percent < 95: return 7
    else: return 8

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        location = data.get("location")
        date_str = data.get("date")

        if not location or not date_str:
            return jsonify({"error": "Missing location or date"}), 400

        # 1️⃣ Get lat/lon from OpenCage
        geo_url = f"https://api.opencagedata.com/geocode/v1/json?q={location}&key={OPENCAGE_API_KEY}"
        geo_resp = requests.get(geo_url)
        if geo_resp.status_code != 200:
            return jsonify({"error": "OpenCage API request failed"}), 500
        geo_data = geo_resp.json()
        if not geo_data["results"]:
            return jsonify({"error": "Location not found"}), 404
        lat = geo_data["results"][0]["geometry"]["lat"]
        lon = geo_data["results"][0]["geometry"]["lng"]

        # 2️⃣ Get NASA POWER daily data
        nasa_url = (
            "https://power.larc.nasa.gov/api/temporal/daily/point"
            f"?parameters=T2M,PS,ALLSKY_SZA,ALBEDO,CLDTOT,RH2M,WS10M"
            f"&start=20170102&end=20250801"
            f"&latitude={lat}&longitude={lon}&format=JSON"
        )
        nasa_resp = requests.get(nasa_url)
        if nasa_resp.status_code != 200:
            return jsonify({"error": f"NASA POWER API request failed: {nasa_resp.text}"}), 500

        nasa_data = nasa_resp.json()
        date_key = date_str.replace("-", "")

        try:
            temp = nasa_data["properties"]["parameter"]["T2M"][date_key]
            pres = nasa_data["properties"]["parameter"]["PS"][date_key]
            sza = nasa_data["properties"]["parameter"]["ALLSKY_SZA"][date_key]
            alb = nasa_data["properties"]["parameter"]["ALBEDO"][date_key]
            cld_percent = nasa_data["properties"]["parameter"]["CLDTOT"][date_key]
            rh = nasa_data["properties"]["parameter"]["RH2M"][date_key]
            wind = nasa_data["properties"]["parameter"]["WS10M"][date_key]
        except KeyError:
            return jsonify({"error": f"No NASA data available for {date_str}"}), 404

        # 3️⃣ Map cloud % to model category
        cloud_type = map_cloud_percent_to_type(cld_percent)

        # 4️⃣ Prepare features for the ML model
        month = datetime.strptime(date_str, "%Y-%m-%d").month
        features = [[month, temp, pres, sza, alb, cloud_type, rh, wind]]

        # 5️⃣ Predict
        prediction = model.predict(features)[0]

        return jsonify({
            "location": location,
            "latitude": lat,
            "longitude": lon,
            "date": date_str,
            "inputs": {
                "month": month,
                "temperature": temp,
                "pressure": pres,
                "solar_zenith_angle": sza,
                "surface_albedo": alb,
                "cloud_type": cloud_type,
                "relative_humidity": rh,
                "wind_speed": wind
            },
            "prediction": prediction
        })

    except Exception as e:
        return jsonify({"error": str(e), "type": type(e).__name__}), 500




if __name__ == "__main__":
    import os

    # Check if running in a production environment (PORT is usually set by hosting services)
    port = int(os.environ.get("PORT", 5000))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"

    # Debug mode only if running locally
    debug_mode = not os.environ.get("PORT")

    app.run(host=host, port=port, debug=debug_mode)





