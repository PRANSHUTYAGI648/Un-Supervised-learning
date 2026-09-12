from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained K-Means model
kmeans = joblib.load("model/model.pkl")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    income = float(request.form["income"])
    spending = float(request.form["spending"])

    # Create input data
    input_data = pd.DataFrame(
        [[income, spending]],
        columns=["Annual Income (k$)", "Spending Score (1-100)"]
    )

    # Predict customer cluster
    cluster = kmeans.predict(input_data)[0]

    return render_template(
        "index.html",
        prediction=f"Customer belongs to Cluster {cluster}"
    )


if __name__ == "__main__":
    app.run(debug=True)