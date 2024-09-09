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


# def get_response(sentence):
    # global current_question, current_answer
    
    # # Keep the original input for answer comparison
    # raw_input = sentence

    # # Tokenize and process the input for intent recognition
    # sentence = tokenize(sentence)
    # x = bag_of_words(sentence, all_words)
    # x = x.reshape(1, x.shape[0])
    # x = torch.from_numpy(x).to(device)

    # output = model(x)
    # _, predicted = torch.max(output, dim=1)
    # tag = tags[predicted.item()]

    # probs = torch.softmax(output, dim=1)
    # prob = probs[0][predicted.item()]

    # # Use the raw_input (unprocessed sentence) to compare with the current_answer
    # if current_question:
    #     if raw_input.strip().lower() in current_answer.lower():
    #         print(f"{bot_name}: Correct! The answer is {current_answer}.")
    #     else:
    #         print(f"{bot_name}: Incorrect. The correct answer is {current_answer}.")
    #     # Reset after validation
    #     current_question = None
    #     current_answer = None
        

    # if prob.item() > 0.75:
    #     for intent in intents["intents"]:
    #         if tag == intent["tag"]:
    #             # Check if this intent has a trivia question and answer
    #             if 'answers' in intent:
    #                 idx = random.randint(0, len(intent['responses']) - 1)
    #                 current_question = intent['responses'][idx]
    #                 current_answer = intent['answers'][idx].lower()  # Store the correct answer
    #                 return {current_question}
    #             else:
    #                 return random.choice(intent['responses'])
    # else:
    #     return "I do not understand..."


# while True:

#     # Keep the original input for answer comparison
#     raw_input = sentence

#     # Tokenize and process the input for intent recognition
#     sentence = tokenize(sentence)
#     x = bag_of_words(sentence, all_words)
#     x = x.reshape(1, x.shape[0])
#     x = torch.from_numpy(x).to(device)

#     output = model(x)
#     _, predicted = torch.max(output, dim=1)
#     tag = tags[predicted.item()]

#     probs = torch.softmax(output, dim=1)
#     prob = probs[0][predicted.item()]

#     # Use the raw_input (unprocessed sentence) to compare with the current_answer
#     if current_question:
#         if raw_input.strip().lower() in current_answer.lower():
#             print(f"{bot_name}: Correct! The answer is {current_answer}.")
#         else:
#             print(f"{bot_name}: Incorrect. The correct answer is {current_answer}.")
#         # Reset after validation
#         current_question = None
#         current_answer = None
#         continue

#     if prob.item() > 0.75:
#         for intent in intents["intents"]:
#             if tag == intent["tag"]:
#                 # Check if this intent has a trivia question and answer
#                 if 'answers' in intent:
#                     idx = random.randint(0, len(intent['responses']) - 1)
#                     current_question = intent['responses'][idx]
#                     current_answer = intent['answers'][idx].lower()  # Store the correct answer
#                     print(f"{bot_name}: {current_question}")
#                 else:
#                     print(f"{bot_name}: {random.choice(intent['responses'])}")
#     else:
#         print(f"{bot_name}: I do not understand...")
