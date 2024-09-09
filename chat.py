import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "TriviaPal"
# print("Let's chat! type 'quit' to exit")
# Variables to store the current question and answer
current_question = None
current_answer = None

def get_response(msg):
    global current_question, current_answer
    
    # Tokenize the input message
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    # Predict the intent
    output = model(X)
    _, predicted = torch.max(output, dim=1)

    # Get the predicted tag
    tag = tags[predicted.item()]

    # Calculate the probability
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # Step 3: If the user has been asked a question, validate their response
    if current_question and current_answer:
        # Check if the user's response matches the answer
        if msg.strip().lower() in current_answer.lower():
            response = f"Correct! The answer is {current_answer}."
        else:
            response = f"Incorrect. The correct answer is {current_answer}."
        
        # Reset after answering
        current_question = None
        current_answer = None
        return response

    # Step 1: Check if the intent is a question and validate the next response
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                
                # Step 2: If the intent has an 'answers' key, it's a trivia question
                if 'answers' in intent:
                    idx = random.randint(0, len(intent['responses']) - 1)
                    
                    # Store the current question and answer
                    current_question = intent['responses'][idx]
                    current_answer = intent['answers'][idx]

                    # Return the question to the user
                    return current_question
                
                # If it's not a trivia question, return a normal response
                return random.choice(intent['responses'])
    
    return "I do not understand..."