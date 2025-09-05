from flask import Flask, render_template, request, redirect, url_for, flash,jsonify,session
from google import genai
import sqlite3
import os
import secrets
from dotenv import load_dotenv
from werkzeug.security import check_password_hash
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
from datetime import datetime
import pytz

ist=pytz.timezone('Asia/Kolkata')
current_time = datetime.now(ist).strftime('%Y-%m-%d %H:%M')

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
    

# @app.route("/api/analyze_mood", methods=["POST"])
# def analyze_mood():
#     data = request.get_json()
#     entry = data.get("entry", "")

#     if not entry.strip():
#         return jsonify({"error": "Empty journal entry"}), 400

#     scores = analyzer.polarity_scores(entry)
#     compound = scores['compound']  

#     mood_score = int((compound + 1) * 50)

#     if mood_score < 30:
#         mood_label = "Very Low"
#     elif 30 <= mood_score < 60:
#         mood_label = "Neutral"
#     elif 60 <= mood_score < 80:
#         mood_label = "Good"
#     else:
#         mood_label = "Blissful"

#     return jsonify({
#         "mood_score": mood_score,
#         "mood_label": mood_label,
#         "raw_scores": scores  # optional debug
#     })


@app.route("/api/analyze_mood", methods=["POST"])
def analyze_mood():
    data = request.get_json()
    entry = data.get("entry", "")
    if not entry.strip():
        return jsonify({"error": "Empty journal entry"}), 400
    
    scores = analyzer.polarity_scores(entry)
    compound = scores['compound']  
    mood_score = int((compound + 1) * 50)
    
    # Enhanced mood categorization with 5 levels
    if mood_score < 20:
        mood_label = "Very Low"
    elif 20 <= mood_score < 40:
        mood_label = "Low"
    elif 40 <= mood_score < 60:
        mood_label = "Neutral"
    elif 60 <= mood_score < 80:
        mood_label = "Good"
    else:  # 80 <= mood_score <= 100
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
                "INSERT INTO journal_entries (user_id, entry, ai_response,mood,created_at) VALUES (?, ?, ?,?,?)",
                (session["user_id"], entry, ai_response,mood,current_time)
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


@app.route('/api/mood-calendar/<int:year>/<int:month>')
def get_mood_calendar(year, month):
    """
    Get mood data for calendar display based on dominant mood per date
    Returns: {date: emoji} mapping for the specified month/year
    """
    try:
        # Use your existing database connection function
        conn = get_db_connection()
        cursor = conn.cursor()
       
        # Since you're already saving IST time, no timezone conversion needed
        cursor.execute('''
            SELECT
                DATE(created_at) as entry_date,
                mood
            FROM journal_entries
            WHERE user_id = ?
            AND strftime('%Y', created_at) = ?
            AND strftime('%m', created_at) = ?
            AND mood IS NOT NULL
            ORDER BY entry_date, created_at
        ''', (session['user_id'], str(year), f"{month:02d}"))
       
        results = cursor.fetchall()
        conn.close()
       
        # Define mood hierarchy and emojis
        mood_emojis = {
            'blissful': '😁',
            'excellent': '😁',
            'very_happy': '😁',
            'good': '😊',
            'happy': '😊',
            'positive': '😊',
            'neutral': '😐',
            'okay': '😐',
            'average': '😐',
            'low': '😔',
            'sad': '😔',
            'down': '😔',
            'very_low': '😣',
            'very low': '😣',
            'terrible': '😣',
            'awful': '😣'
        }
       
        # Group entries by date and count mood frequencies
        date_moods = {}
        for row in results:  # Changed: handle sqlite3.Row objects
            day = int(row['entry_date'].split('-')[2])  # Extract day from YYYY-MM-DD
           
            if day not in date_moods:
                date_moods[day] = []
           
            # Normalize mood to lowercase for comparison
            normalized_mood = row['mood'].lower().strip()  # Changed: use row['mood']
            date_moods[day].append(normalized_mood)
       
        # Calculate dominant mood for each date
        calendar_data = {}
        for day, moods in date_moods.items():
            # Count frequency of each mood
            mood_counter = Counter(moods)
           
            # Get the most common mood
            dominant_mood = mood_counter.most_common(1)[0][0]
           
            # Get emoji for dominant mood
            emoji = mood_emojis.get(dominant_mood, '😐')  # Default to neutral
            calendar_data[day] = emoji
           
        return jsonify({
            'success': True,
            'data': calendar_data,
            'month': month,
            'year': year,
            'summary': {
                'total_entries_found': len(results),
                'dates_with_entries': list(date_moods.keys())
            }
        })
       
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
#########################################################




@app.route('/api/mood-calendar', methods=['POST'])
def get_mood_calendar_post():
    """
    Get mood data for calendar display based on dominant mood per date
    POST method for testing with user_id in request body
    
    Request Body:
    {
        "user_id": 1,
        "year": 2025,
        "month": 8
    }
    
    Returns: {date: emoji} mapping for the specified month/year
    """
    try:
        # Get data from request body
        data = request.get_json()
        
        # Validate required fields
        if not data:
            return jsonify({
                'success': False,
                'error': 'Request body is required'
            }), 400
            
        user_id = data.get('user_id')
        year = data.get('year', datetime.now().year)
        month = data.get('month', datetime.now().month)
        
        if not user_id:
            return jsonify({
                'success': False,
                'error': 'user_id is required'
            }), 400
            
        # Validate year and month
        if not (2020 <= year <= 2030):
            return jsonify({
                'success': False,
                'error': 'Invalid year. Must be between 2020-2030'
            }), 400
            
        if not (1 <= month <= 12):
            return jsonify({
                'success': False,
                'error': 'Invalid month. Must be between 1-12'
            }), 400
        
        # Connect to database using your existing function
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='journal_entries'")
        table_exists = cursor.fetchone()
        
        if not table_exists:
            conn.close()
            return jsonify({
                'success': False,
                'error': 'journal_entries table not found in database'
            }), 500
        cursor.execute('''
            SELECT 
                DATE(created_at) as entry_date,
                mood
            FROM journal_entries 
            WHERE user_id = ? 
            AND strftime('%Y', created_at) = ?
            AND strftime('%m', created_at) = ?
            AND mood IS NOT NULL
            ORDER BY entry_date, created_at
        ''', (user_id, str(year), f"{month:02d}"))
        
        results = cursor.fetchall()
        conn.close()
        
        # Define mood hierarchy and emojis
        mood_emojis = {
            'blissful': '😁',
            'excellent': '😁', 
            'very_happy': '😁',
            'good': '😊',
            'happy': '😊',
            'positive': '😊',
            'neutral': '😐',
            'okay': '😐',
            'average': '😐',
            'low': '😔',
            'sad': '😔',
            'down': '😔',
            'very_low': '😣',
            'very low': '😣',
            'terrible': '😣',
            'awful': '😣'
        }
        
        # Group entries by date and collect all moods for each date
        date_moods = {}
        for row in results:
            day = int(row['entry_date'].split('-')[2])  # Extract day from YYYY-MM-DD
            
            if day not in date_moods:
                date_moods[day] = []
            
            # Normalize mood to lowercase for comparison
            normalized_mood = row['mood'].lower().strip()
            date_moods[day].append(normalized_mood)
        
        # Calculate dominant mood for each date
        calendar_data = {}
        mood_details = {}  # For debugging
        
        for day, moods in date_moods.items():
            # Count frequency of each mood
            mood_counter = Counter(moods)
            
            # Get the most common mood (if tie, returns first one)
            dominant_mood = mood_counter.most_common(1)[0][0]
            dominant_count = mood_counter.most_common(1)[0][1]
            
            # Get emoji for dominant mood
            emoji = mood_emojis.get(dominant_mood, '😐')  # Default to neutral
            calendar_data[day] = emoji
            
            # Store details for debugging
            mood_details[day] = {
                'dominant_mood': dominant_mood,
                'count': dominant_count,
                'total_entries': len(moods),
                'all_moods': dict(mood_counter)
            }
            
        return jsonify({
            'success': True,
            'data': calendar_data,
            'month': month,
            'year': year,
            'summary': {
                'total_entries_found': len(results),
                'dates_with_entries': list(date_moods.keys()),
                'total_dates_with_moods': len(calendar_data)
            },
            'mood_details': mood_details  # Remove this in production
        })
        
    except sqlite3.Error as db_error:
        return jsonify({
            'success': False,
            'error': f'Database error: {str(db_error)}'
        }), 500
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

if __name__ == "__main__":
    app.run(debug=True)
