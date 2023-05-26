from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Define training data
training_data = [
    ("What's the weather today?", "weather"),
    ("Tell me a joke.", "joke"),
    ("How to bake a cake?", "recipe"),
    ("What's the capital of France?", "capital"),
    # Add more training examples
]

# Define the AI engine pipeline
pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", LogisticRegression())
])

# Train the AI engine
X_train = [data[0] for data in training_data]
y_train = [data[1] for data in training_data]
pipeline.fit(X_train, y_train)

# Main conversation loop
while True:
    user_input = input("User: ")

    # Get the predicted category from the AI engine
    predicted_category = pipeline.predict([user_input])[0]

    # Handle the predicted category and generate a response
    if predicted_category == "weather":
        response = "The weather today is sunny."
    elif predicted_category == "joke":
        response = "Why don't scientists trust atoms? Because they make up everything!"
    elif predicted_category == "recipe":
        response = "Here's a simple cake recipe: ..."
    elif predicted_category == "capital":
        response = "The capital of France is Paris."
    else:
        response = "I'm sorry, I don't understand."

    print("ChatBot:", response)

