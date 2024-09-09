# trivia-pal-bot
Simple chatbot that knows a tiny bit of trivia

## Installation Steps: 
1. Clone the repository:
```bash
git clone https://github.com/shilpa-bo/trivia-pal-bot.git
```
2. Navigate to project directory and active an environment (I am using Conda)
```bash
cd trivia-pal-bot
conda create --name trivia-env
conda activate trivia-env
```
3. Install PyTorch (depends on system configuration) and other dependencies
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install -r requirements.txt
```
4. Run the chatbot!
```bash
python3 app.py
```

## About this project:
### How it was built:
The Trivia Pal Bot was devoloped with the help of PyTorch and several key machine learning and natural language processing (NLP) techniques. Below is a breakdown of how the chatbot works and the underlying technologies.

**PyTorch Neural Network**
The core of the chatbot is a neural network built using PyTorch. The network classifies user inputs into various intents (greeting, trivia questions, goodbye) using a **feedforward neural netword**
**Network Structure**
- **Input Layer**: The input size corresponds to the number of features extracted from the users input (using tokenization and bag of words)
- **Hidden Layers**: Hidden layers that utilitze **ReLU activation functions**, this introduces non-linearity which allows the network to learn more complex patterns
- **Output Layer**: The output size corresponds to the number of intents (categories of responses) </br>

**NLP Pipeline** </br>
The chatbot processes the input using the following NLP steps: </br>
**Tokenization** -> **Stemming** -> **Bag of Words** -> **Intent Classification** -> **Response Generation**


### The chatbot in action:
<img width="1313" alt="image" src="https://github.com/user-attachments/assets/43ab847f-8904-45c6-be8f-7cf135b313bf">


