
# run


from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # ← This enables CORS for all origins

# Load your model and columns once
model = joblib.load("final_model.pkl")
col_names = joblib.load("column_names.pkl")



@app.route("/predict", methods=["POST"])
def predict():
    feat_data = request.get_json()
    print("Received JSON data:", feat_data)

    # Fix capitalization
    feat_data_fixed = {}
    for col in col_names:
        for key in feat_data:
            if key.lower() == col.lower():
                feat_data_fixed[col] = feat_data[key]
                break
        else:
            return jsonify({"error": f"Missing value for column: {col}"}), 400

    df = pd.DataFrame([feat_data_fixed])
    df = df.reindex(columns=col_names)

    print("DataFrame going into model:\n", df)

    if df.isnull().values.any():
        return jsonify({"error": "Input contains NaN values."}), 400

    prediction = list(model.predict(df))
    return jsonify({"prediction": str(prediction)})


if __name__ == '__main__':
    app.run(port=5000, debug=True)