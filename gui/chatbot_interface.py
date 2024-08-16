import sys
import os
import tkinter as tk
from tkinter import scrolledtext, font
import time
from PIL import Image, ImageTk

# Ensure the parent directory of 'scripts' is in the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.chatbot_response import get_response

# Function to simulate bot typing
def bot_typing_indicator():
    chat_window.insert(tk.END, "Bot is typing...\n")
    chat_window.update_idletasks()
    time.sleep(1)
    chat_window.delete('end-2l', 'end-1l')  # Remove the typing indicator

def send_message():
    user_message = message_entry.get()
    if user_message.strip():
        chat_window.insert(tk.END, f"You ({time.strftime('%H:%M:%S')}): {user_message}\n", 'user')
        message_entry.delete(0, tk.END)
        bot_typing_indicator()
        bot_response = get_response(user_message)
        chat_window.insert(tk.END, f"Bot ({time.strftime('%H:%M:%S')}): {bot_response}\n", 'bot')

# Create the main window
root = tk.Tk()
root.title("AI ChatBot")
root.configure(bg='#2c3e50')

# Custom font
custom_font = font.Font(family="Helvetica", size=12)

# Chat window with custom styles
chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='normal', bg='#34495e', fg='#ecf0f1', font=custom_font)
chat_window.pack(padx=10, pady=10)
chat_window.tag_config('user', foreground='#1abc9c', font=custom_font)
chat_window.tag_config('bot', foreground='#e74c3c', font=custom_font)

# Profile picture for bot and user
send_icon_path = "F:/Python Projects/chatbot_project/gui/icons/send_icon.png"
mic_icon_path = "F:/Python Projects/chatbot_project/gui/icons/mic_icon.png"
user_icon_path = "F:/Python Projects/chatbot_project/gui/icons/user_icon.png"
bot_icon_path = "F:/Python Projects/chatbot_project/gui/icons/bot_icon.png"

send_icon = ImageTk.PhotoImage(Image.open(send_icon_path))
mic_icon = ImageTk.PhotoImage(Image.open(mic_icon_path))
user_icon = ImageTk.PhotoImage(Image.open(user_icon_path))
bot_icon = ImageTk.PhotoImage(Image.open(bot_icon_path))

# Entry widget for user input with custom styles
message_entry = tk.Entry(root, width=80, font=custom_font, bg='#95a5a6', fg='#2c3e50')
message_entry.pack(padx=10, pady=10)
message_entry.bind("<Return>", lambda event: send_message())

# Send button with icon
send_button = tk.Button(root, image=send_icon, command=send_message, bg='#1abc9c', borderwidth=0)
send_button.pack(padx=10, pady=10)

# Voice input button (just a placeholder for functionality)
voice_button = tk.Button(root, image=mic_icon, command=lambda: print("Voice input clicked!"), bg='#1abc9c', borderwidth=0)
voice_button.pack(padx=10, pady=10)

# Make the window resizable
root.resizable(True, True)

# Run the main loop
root.mainloop()