import pickle
import openai
from flask import Flask, request, jsonify
import os  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import logging

class med:
    def __init__(self, medi, compo):
        self.medi = medi
        self.compo = compo
        self.next = None

class data:
    def __init__(self):
        self.head = None
    
    def __iter__(self):
        current = self.head
        while current:
            yield current
            current = current.next
    
    def insert(self, medi, compo):
        new_node = med(medi, compo)
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
    
    def display(self):
        current = self.head
        while current:
            print(f"Medicines: {current.medi}, Data: {current.compo}")
            current = current.next

app = Flask(__name__)

openai.api_key = "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"

# store chat messages
messages = [{"role": "system", "content": "You are a pharmacist"}]

# data to a file
def save_data_to_file(data):
    with open("data.pkl", "wb") as file:
        pickle.dump(data, file)

# data from a file
def load_data_from_file():
    if os.path.exists("data.pkl"):
        with open("data.pkl", "rb") as file:
            return pickle.load(file)
    else:
        return data()  

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@app.route('/')
def home():
    return "ChatGPT Flask API"

@app.route('/chat', methods=['POST'])
def chat():
    database = load_data_from_file()
    user_message = request.json.get('message')
    if user_message:
        # Check if the message already exists in the data structure
        current = database.head
        while current:
            if current.medi == user_message:
                return jsonify({"reply": current.compo})
            current = current.next
        # If the message is not found in the data structure, proceed with GPT call
        user_message1 = "please provide final composition of when these two tablets are taken together:" + user_message + 'Note: just give the names of final active components nothing else!'
        messages.append({"role": "user", "content": user_message1})
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            if response:
                reply = response["choices"][0]["message"]["content"]
                messages.append({"role": "assistant", "content": reply})
                database.insert(user_message, reply)
                database.display()
                save_data_to_file(database) 
                return jsonify({"reply": reply})
            else:
                return jsonify({"error": "An error occurred during the API call."})
        except Exception as e:
            return jsonify({"error": f"An error occurred during the API call: {e}"})
    else:
        return jsonify({"error": "No message provided."})

# Integrate model training route
@app.route('/train_model', methods=['POST'])
def train_model():
    database = load_data_from_file()
    if not database.head or not database.head.next:
        return jsonify({"error": "Insufficient data to train the model."})

    X = [med.medi for med in database]
    y = [med.compo for med in database]

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    except ValueError as e:
        return jsonify({"error": str(e)})

    # Define a pipeline for text classification
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),  # Convert text to numerical features using TF-IDF
        ('clf', LogisticRegression()),  # Train a Logistic Regression classifier
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info("Accuracy: %f", accuracy)
    logger.info("Classification Report:\n%s", classification_report(y_test, y_pred))

    # Save the trained model
    with open("model.pkl", "wb") as file:
        pickle.dump(pipeline, file)

    return jsonify(accuracy,classification_report(y_test, y_pred))


if __name__ == "__main__":
    app.run(debug=True)
