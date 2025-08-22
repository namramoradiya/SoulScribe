import sqlite3

# Step 1: Connect to database (will create 'soulscribe.db' if it doesn't exist)
conn = sqlite3.connect('soulscribe.db')

# Step 2: Create a cursor to execute SQL commands
cursor = conn.cursor()

# Step 3: Create table for registration
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS journal_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    entry TEXT NOT NULL,
    ai_response TEXT,  
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
)
''')


# Step 4: Save (commit) changes and close connection
conn.commit()
conn.close()

print("Database and table created successfully.")
