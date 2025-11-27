from flask import Flask, render_template, request
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
app = Flask(__name__)

# Load Model & Vector
vector = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("phishing.pkl", 'rb'))


@app.route("/")
def home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    url_input = request.form['url']

    # ----- 🔐 PREDICTION -----
    features = vector.transform([url_input])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    confidence = np.max(probability) * 100

    # ----- 🧠 SHAP EXPLANATION BAR CHART -----


    # Set class explicitly
    model.classes_ = np.array(['bad', 'good'])

    # Small fixed background URLs
    background_urls = ["http://example.com", "https://google.com", "https://github.com"]
    background_vec = vector.transform(background_urls)

    mean_dense = background_vec.mean(axis=0).A1  # mean as dense
    coef_dense = model.coef_.reshape(-1)

    # Input dense vector
    input_dense = features.toarray().reshape(-1)

    # Manual SHAP Calculation (Linear)
    shap_values = (input_dense - mean_dense) * coef_dense

    # Top 12 influence words
    feature_names = vector.get_feature_names_out()
    top_n = 12
    indices = np.argsort(np.abs(shap_values))[::-1][:top_n]
    top_features = feature_names[indices]
    top_importance = shap_values[indices]

    # Plot
    colors = ['red' if val > 0 else 'blue' for val in top_importance]

    plt.figure(figsize=(9, 4))
    plt.barh(top_features, top_importance, color=colors)
    plt.xlabel("Influence on Prediction (← Phishing | Legitimate →)")
    plt.title("Top URL Features Influencing Prediction")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # Save to static folder
    if not os.path.exists("static"):
        os.makedirs("static")

    shap_img_path = "static/shap_bar.png"
    plt.savefig(shap_img_path)
    plt.close()

    # Final result text
    result = "Phishing Site ⚠️" if prediction == "bad" else "Legitimate Site ✅"

    return render_template(
        'result.html',
        url=url_input,
        result=result,
        confidence=round(confidence, 2),
        shap_img_path=shap_img_path
    )


if __name__ == "__main__":
    app.run(debug=True)
