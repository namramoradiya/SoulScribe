from flask import Flask, request, jsonify
from textblob import TextBlob

app = Flask(__name__)

@app.route("/analyze_mood", methods=["POST"])
def analyze_mood():
    try:
        data = request.get_json()
        entry = data.get("entry", "")

        if not entry.strip():
            return jsonify({"error": "Empty entry"}), 400

        # 🔹 Simple Sentiment Analysis using TextBlob
        blob = TextBlob(entry)
        polarity = blob.sentiment.polarity  # -1 (negative) → +1 (positive)

        # Convert polarity (-1 to 1) into 0–100 scale
        mood_score = int((polarity + 1) * 50)

        # Mood label based on score
        if mood_score < 30:
            mood_label = "Very Low"
        elif 30 <= mood_score < 60:
            mood_label = "Neutral"
        elif 60 <= mood_score < 80:
            mood_label = "Good"
        else:
            mood_label = "Blissful"

        return jsonify({
            "mood_score": mood_score,
            "mood_label": mood_label
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500




if __name__ == "__main__":
    app.run(debug=True, port=5001)
