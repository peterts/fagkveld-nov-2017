from flask import Flask, request, jsonify
import os
import joblib
from config import MODEL_FILE_NAME
from model_fitter import fit_model

app = Flask(__name__)

model = None

@app.route("/api/v1/predict", methods=["POST"])
def predict():
    documents = request.get_json().get("documents")
    predictions = model.predict(documents)
    return jsonify({"predictions": list(predictions)})

if __name__ == "__main__":
    if not os.path.exøists(MODEL_FILE_NAME):
        print("Fitting model")
        fit_model()
    model = joblib.load(MODEL_FILE_NAME)
    print("Starting app")
    app.run(host="localhost", port=8081)

