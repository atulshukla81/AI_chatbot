import sqlite3
from scripts.chatbot_response import predict_class, get_response as generate_response

# Connect to SQLite database
conn = sqlite3.connect('../database/chatbot.db')
c = conn.cursor()

# Create a table to store interactions
c.execute('''CREATE TABLE IF NOT EXISTS interactions (user_input TEXT, bot_response TEXT)''')
conn.commit()

def save_interaction(user_input, bot_response):
    c.execute("INSERT INTO interactions (user_input, bot_response) VALUES (?, ?)", (user_input, bot_response))
    conn.commit()

def get_response(user_input):
    ints = predict_class(user_input)  # Assuming predict_class returns the predicted intent
    response = generate_response(ints)  # Assuming generate_response creates a response based on the intent
    save_interaction(user_input, response)  # Save the interaction to the database
    return response
