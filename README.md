# IntelliChat
This repository contains a simple ChatBot implemented in Python using the TensorFlow library. The ChatBot is trained to understand and respond to various user inputs based on pre-defined intents.

## Files
**chatbot.py:** This file contains the code for the ChatBot itself. It loads a pre-trained model, processes user input, predicts the intent, and generates a response.

**training.py:** This file is used for training the ChatBot. It processes the intents specified in the intents.json file, tokenizes the patterns, and creates a bag of words representation for training a neural network.

**intents.json:** This JSON file defines the intents for the ChatBot. Each intent consists of a tag, a list of patterns, and corresponding responses. This file is used for both training and responding to user inputs.

## Usage
#### Training the ChatBot
Before using the ChatBot, you need to train it using the training.py script. This script processes the intents, tokenizes patterns, and trains a neural network. After training, it saves the model and necessary files (words.pkl and classes.pkl) for later use.

#### Running the ChatBot
Once the ChatBot is trained, you can run the chatbot.py script to interact with it. The ChatBot will continuously prompt for user input, predict the intent, and generate a response.

## File Descriptions
**words.pkl:** Pickle file containing the unique words obtained from the training data.

**classes.pkl:** Pickle file containing the unique classes (intents) obtained from the training data.

**chatbot_model.h5:** The trained neural network model saved in the Hierarchical Data Format (HDF5).

## Intents Configuration
You can customize the behavior of the ChatBot by modifying the intents.json file. Add new intents with associated patterns and responses to enhance the ChatBot's capabilities.

## Dependencies
Ensure that you have the necessary dependencies installed before running the scripts. You can install them using:
pip install -r requirements.txt

## Notes
The ChatBot uses the natural language processing library NLTK for tokenization and lemmatization.

The neural network architecture consists of dense layers with ReLU activation for processing the bag of words representation.

## Acknowledgments
The ChatBot implementation is inspired by various tutorials and resources on natural language processing and chatbot development.

Feel free to experiment, enhance, and adapt the code to suit your specific use case!
