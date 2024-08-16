# AI ChatBot Project

This project is a simple AI chatbot built using Python, Keras, NLTK, Tkinter, JSON, and SQLite.

## Folder Structure
chatbot_project/
│
├── data/
│ ├── intents.json # JSON file containing the chatbot intents
│ ├── words.json # JSON file storing the words used in training
│ ├── classes.json # JSON file storing the classes of intents
│
├── models/
│ ├── chatbot_model.h5 # Trained Keras model for the chatbot
│
├── database/
│ ├── chatbot.db # SQLite database storing user interactions
│
├── scripts/
│ ├── train_chatbot.py # Script to train the chatbot model
│ ├── nlp_utils.py # NLP utilities (tokenization, bag of words, etc.)
│ ├── chatbot_response.py # Script for generating chatbot responses
│ ├── save_interaction.py # Script to save interactions to the database
│
├── gui/
│ ├── chatbot_interface.py # Tkinter GUI for the chatbot
│
└── README.md # Documentation or instructions for your project
Install the dependencies:
pip install -r requirements.txt


Train the Chatbot Model:
cd scripts
python train_chatbot.py


Run the Chatbot Interface:
After training the model, navigate to the gui directory and run the chatbot_interface.py script to start the Tkinter interface.
cd ../gui
python chatbot_interface.py


Interact with the Chatbot:

Use the interface to interact with the chatbot. All interactions will be stored in the SQLite database.


The intents.json file can be modified to add more patterns and responses.
The SQLite database is used to store interactions for further analysis.

### Running Steps

1. **Clone the Repository**:
   - If you have set up version control, clone the repository. Otherwise, manually create the directory structure on your machine.

2. **Install the Dependencies**:
   - Ensure you have Python and pip installed. Install the necessary Python packages.
   ```bash
   pip install nltk keras tensorflow sqlite3 tkinter

Train the Chatbot Model:

Run the train_chatbot.py script to train your model and generate the necessary files (words.json, classes.json, and chatbot_model.h5).
python scripts/train_chatbot.py


Run the Chatbot Interface:

Start the chatbot GUI using Tkinter.

Run the Chatbot Interface:

Start the chatbot GUI using Tkinter.