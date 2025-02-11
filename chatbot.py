from transformers import pipeline
import random

class ChatBot:
    def __init__(self):
        # Ensure Hugging Face uses PyTorch
        self.intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", framework="pt")
        self.sentiment_analyzer = pipeline("sentiment-analysis", framework="pt")

        # Define possible intents and responses
        self.intents = {
            "greeting": ["Hi there! How can I help you today?", "Hello! What can I do for you?", "Hey! How can I assist?"],
            "pricing": ["Our prices vary depending on the service. Let me know if you want specifics.", "I can help you explore our pricing plans. What exactly are you interested in?"],
            "support": ["I'm here to help. What do you need assistance with?", "Sure! What seems to be the issue?", "Could you provide more details so I can assist you?"],
            "farewell": ["Goodbye! Have a great day!", "Take care! Feel free to reach out anytime.", "Bye for now! Stay safe."]
        }

        self.default_response = "I'm sorry, I didn't understand that. Could you rephrase or provide more details?"
        self.confidence_threshold = 0.3  # Minimum confidence level to accept an intent

    def detect_intent(self, message):
        if not message.strip():
            return None  # Handle empty input gracefully

        # Define possible intents
        possible_labels = list(self.intents.keys())

        # Classify the intent using zero-shot classification
        classification = self.intent_classifier(message, possible_labels)
        top_intent = classification["labels"][0]
        confidence = classification["scores"][0]

        print(f"Detected Intent: {top_intent}, Confidence: {confidence}")

        # Return intent if confidence meets the threshold
        return top_intent if confidence >= self.confidence_threshold else None

    def analyze_sentiment(self, message):
        if not message.strip():
            return "neutral"  # Return neutral for empty messages

        # Analyze sentiment using Hugging Face's pipeline
        sentiment = self.sentiment_analyzer(message)[0]
        print(f"Detected Sentiment: {sentiment['label']}, Confidence: {sentiment['score']}")

        return sentiment["label"].capitalize()  # Return sentiment with proper casing (Positive, Negative, Neutral)

    def get_response(self, message):
        # Detect the intent and analyze sentiment
        intent = self.detect_intent(message)
        sentiment = self.analyze_sentiment(message)

        # Respond based on the intent and adjust for negative sentiment if necessary
        if intent:
            if sentiment.lower() == "negative":
                return "I’m sorry to hear that. Could you provide more details so I can assist you better?", sentiment
            else:
                return random.choice(self.intents[intent]), sentiment

        # Fallback response if intent is not detected
        return self.default_response, sentiment
