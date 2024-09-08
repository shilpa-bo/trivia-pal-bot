import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
current_question = None
current_answer = None


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
print("Let's chat! type 'quit' to exit")

def get_response(sentence):
     # Keep the original input for answer comparison
    raw_input = sentence

    # Tokenize and process the input for intent recognition
    sentence = tokenize(sentence)
    x = bag_of_words(sentence, all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x).to(device)

    output = model(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # Use the raw_input (unprocessed sentence) to compare with the current_answer
    if current_question:
        if raw_input.strip().lower() in current_answer.lower():
            print(f"{bot_name}: Correct! The answer is {current_answer}.")
        else:
            print(f"{bot_name}: Incorrect. The correct answer is {current_answer}.")
        # Reset after validation
        current_question = None
        current_answer = None
        

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                # Check if this intent has a trivia question and answer
                if 'answers' in intent:
                    idx = random.randint(0, len(intent['responses']) - 1)
                    current_question = intent['responses'][idx]
                    current_answer = intent['answers'][idx].lower()  # Store the correct answer
                    return {current_question}
                else:
                    return random.choice(intent['responses'])
    else:
        return "I do not understand..."


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
