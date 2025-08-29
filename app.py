from flask import Flask, render_template, request, redirect, url_for, flash,jsonify,session
from google import genai
import sqlite3
import os
import secrets
from dotenv import load_dotenv
from werkzeug.security import check_password_hash
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(32)  # Add this line

def get_db_connection():
    conn = sqlite3.connect("soulscribe.db")
    conn.row_factory = sqlite3.Row
    return conn

@app.route("/")
def journal():
    return render_template("landing.html")


from werkzeug.security import generate_password_hash

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        # ✅ Hash the password before saving
        hashed_password = generate_password_hash(password)

        try:
            with sqlite3.connect("soulscribe.db", timeout=10) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                    (username, email, hashed_password)
                )
                conn.commit()
            flash("Account created successfully!", "success")
            return redirect(url_for("login"))

        except sqlite3.IntegrityError:
            flash("Username or email already exists.", "danger")

    return render_template("register.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        with sqlite3.connect("soulscribe.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, username, email, password FROM users WHERE email = ?", (email,))
            user = cursor.fetchone()

        if user and check_password_hash(user[3], password):  
            # user[3] = password hash from DB
            session["user_id"] = user[0]
            session["username"] = user[1]
            flash("Login successful!", "success")
            return redirect(url_for("index"))
        else:
            flash("Invalid email or password.", "danger")

    return render_template("login.html")

@app.route("/index")
def index():
    if "user_id" not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for("login"))

    return render_template("index.html", username=session.get("username"))



load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# app = Flask(__name__)

# Initialize Gemini Client
client = genai.Client(api_key=API_KEY)

# Create a chat session
chat = client.chats.create(
    model="gemini-1.5-flash",
    history=[
        {
            "role": "user",
            "parts": [
                {"text": (
                    "You are SoulScribe, a warm and empathetic journaling companion. "
                    "Always acknowledge the user's feelings first, reflect them back, "
                    "and offer 1–2 gentle, supportive suggestions. Keep responses under 4 sentences."
                )}
            ]
        },
        {"role": "model", "parts": [{"text": "Understood. I’ll keep my tone warm, empathetic, and concise."}]}
    ]
)



@app.route("/api/journal", methods=["POST"])
def journal_api():
    data = request.get_json()
    entry = data.get("entry", "")
    if not entry:
        return jsonify({"error": "No journal entry provided"}), 400
    
    try:
        response = chat.send_message(entry)
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# @app.route("/api/journal", methods=["POST"])
# def insert_entry():
#     try:
#         data = request.get_json()
#         user_id = data.get("user_id")
#         entry_text = data.get("entry_text")
#         soulscribe_response = data.get("soulscribe_response")

#         if not user_id or not entry_text or not soulscribe_response:
#             return jsonify({"error": "Missing required fields"}), 400

#         conn = get_db_connection()
#         cursor = conn.cursor()
#         cursor.execute(
#             """
#             INSERT INTO journal_entries (user_id, entry_text, soulscribe_response)
#             VALUES (?, ?, ?)
#             """,
#             (user_id, entry_text, soulscribe_response),
#         )
#         conn.commit()
#         conn.close()

#         return jsonify({"message": "Journal entry saved successfully"}), 201

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
    

@app.route("/api/analyze_mood", methods=["POST"])
def analyze_mood():
    data = request.get_json()
    entry = data.get("entry", "")

    if not entry.strip():
        return jsonify({"error": "Empty journal entry"}), 400

    scores = analyzer.polarity_scores(entry)
    compound = scores['compound']  

    mood_score = int((compound + 1) * 50)

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
        "mood_label": mood_label,
        "raw_scores": scores  # optional debug
    })

@app.route("/api/save_entry", methods=["POST"])
def save_entry():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json() or {}
    entry = (data.get("entry") or "").strip()
    ai_response = (data.get("ai_response") or "").strip()
    mood=data.get("mood","")

    if not entry or not ai_response:
        return jsonify({"error": "Missing entry or ai_response"}), 400

    try:
        with sqlite3.connect("soulscribe.db") as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO journal_entries (user_id, entry, ai_response,mood) VALUES (?, ?, ?,?)",
                (session["user_id"], entry, ai_response,mood)
            )
            entry_id = cur.lastrowid
            conn.commit()
        return jsonify({"ok": True, "entry_id": entry_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

# @app.route("/api/entries", methods=["GET"])
# def get_entries():
#     if "user_id" not in session:
#         return jsonify({"error": "Unauthorized"}), 401

#     try:
#         with sqlite3.connect("soulscribe.db") as conn:
#             cur = conn.cursor()
#             cur.execute("SELECT * FROM journal_entries WHERE user_id = ?", (session["user_id"],))
#             entries = cur.fetchall()

#         return jsonify([dict(entry) for entry in entries])
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
    
@app.route('/history')
def history():
    return render_template('history.html')


@app.route('/calender')
def calender():
    return render_template('calender.html')


@app.route('/api/history', methods=['GET'])
def get_history():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        with sqlite3.connect("soulscribe.db") as conn:
            conn.row_factory = sqlite3.Row  # This allows us to access columns by name
            cur = conn.cursor()
            
            # Get all entries for the logged-in user, ordered by most recent first
            cur.execute("""
                SELECT id, entry, ai_response, mood, created_at 
                FROM journal_entries 
                WHERE user_id = ? 
                ORDER BY created_at DESC
            """, (session["user_id"],))
            
            entries = cur.fetchall()
            
            # Convert to list of dictionaries
            history_data = []
            for entry in entries:
                history_data.append({
                    "id": entry["id"],
                    "entry": entry["entry"],
                    "ai_response": entry["ai_response"],
                    "mood": entry["mood"],
                    "created_at": entry["created_at"],
                    # Format date for display (optional)
                    "formatted_date": format_date(entry["created_at"]) if entry["created_at"] else ""
                })
            
            return jsonify({
                "success": True,
                "entries": history_data,
                "total_entries": len(history_data)
            })
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Helper function to format dates (optional)
def format_date(date_string):
    """Convert database date string to readable format"""
    try:
        from datetime import datetime
        # Assuming your created_at is in format: YYYY-MM-DD HH:MM:SS
        dt = datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')
        return dt.strftime('%B %d, %Y at %I:%M %p')  # e.g., "January 15, 2024 at 02:30 PM"
    except:
        return date_string  # Return original if formatting fails


if __name__ == "__main__":
    app.run(debug=True)
