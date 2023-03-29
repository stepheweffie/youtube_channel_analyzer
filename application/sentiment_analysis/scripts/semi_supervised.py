import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from preprocess import preprocess_text
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelPropagation
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the labeled dataset
labeled_data = pd.read_csv("labeled_data.csv")

# Split the labeled data into training and testing sets
train_data, test_data = train_test_split(labeled_data, test_size=0.2, random_state=42)


# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(preprocessor=preprocess_text)

# Vectorize the labeled training data
train_vectors = vectorizer.fit_transform(train_data["text"])

# Train a sentiment analysis model using LabelPropagation
lp_model = LabelPropagation()
lp_model.fit(train_vectors, train_data["label"])

# Vectorize the unlabeled data
unlabeled_data = pd.read_csv("unlabeled_data.csv")
unlabeled_vectors = vectorizer.transform(unlabeled_data["text"])

# Use the trained model to make predictions on the unlabeled data
predicted_labels = lp_model.predict(unlabeled_vectors)

# Add the predicted labels to the unlabeled data
unlabeled_data["label"] = predicted_labels

# Combine the labeled and unlabeled data
combined_data = pd.concat([train_data, unlabeled_data])

# Vectorize the combined data
combined_vectors = vectorizer.fit_transform(combined_data["text"])

# Train a new sentiment analysis model using the combined data
new_model = LabelPropagation()
new_model.fit(combined_vectors, combined_data["label"])

# Evaluate the model on the test data
test_vectors = vectorizer.transform(test_data["text"])
test_predictions = new_model.predict(test_vectors)

# Calculate the accuracy of the model
accuracy = sum(test_predictions + test_data["label"]) / len(test_predictions)
print("Accuracy:", accuracy)
