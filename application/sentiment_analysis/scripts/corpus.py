import openai
import nltk
import csv
from nltk.sentiment import SentimentIntensityAnalyzer
from preprocess import preprocess_text
from dotenv import load_dotenv
import os


load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
# Set up OpenAI API key
openai.api_key = api_key

# Define a list of prompts related to diversity, equity, and inclusion
prompts = [
    "What are your thoughts on diversity in the workplace?",
    "How can we promote equity in education?",
    "Why is inclusion important in society?",
    "What are some strategies for improving diversity and inclusion in the tech industry?",
    "What does diversity, equity, and inclusion have to do with mental health?",
    "What does diversity, equity, and inclusion have to do with ethics?",
    "What does diversity, equity, and inclusion have to do with rights in the United States?",
    "What do sources say about professional conduct concerning diversity, equity, and inclusion?",
    "Does the American Psychological Association accept diversity, equity, and inclusion?",
    "Where did the idea of diversity, equity, and inclusion originate?",
    "What are some reasons for ethical consideration of diversity, equity, and inclusion?",
    "Are there federal laws in the United States that support diversity, equity, and inclusion?",
    "Are post secondary institutions beholden to diversity, equity, and inclusion?",
    "Do clinicians have the right to be unethical in the United States?",
    "Is a clinician who is against diversity, equity, and inclusion ethical in the United States?",
    "Would ethical media literacy for clinicians include training on diversity, equity, and inclusion?",
    "Diversity, equity, and inclusion is an ethical focus or an indoctrination?",
    "Familiarize me with aspects about diversity, equity, and inclusion we haven't discussed yet.",
    "I need more information about ethics involving diversity, equity, and inclusion.",
    "Do patients and clients of clinicians in the United States have an expectation of ethics?",
    "What can patients and clients expect from diversity, equity, and inclusion in ethical relationships?",
    "Is there a right-wing conspiracy theory involving diversity, equity, and inclusion?",
    "Is there a prevalence of anti diversity, equity, and inclusion rhetoric online?",
    "Is there a prevalence of pro diversity, equity, and inclusion rhetoric online?"
]

# Initialize the corpus as an empty list
corpus = []


def prompt_loop():
    # Use the OpenAI API to generate responses to each prompt
    for prompt in prompts:
        # Generate a response using the OpenAI API
        response = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5
        )

        # Extract the response text from the API response
        response_text = response.choices[0].text.strip()

        # Add the response text to the corpus
        corpus.append(response_text)


for i in range(5):
    prompt_loop()


# Pre-process the corpusf
preprocessed_corpus = [preprocess_text(text) for text in corpus]

# Use a sentiment analysis tool to assign a sentiment score to each response
sia = SentimentIntensityAnalyzer()
sentences = nltk.sent_tokenize(preprocessed_corpus)
scores = [sia.polarity_scores(sentence)["compound"] for sentence in sentences]

# Label each response as positive, negative, or neutral based on the sentiment score
labels = []
for score in scores:
    if score > 0.2:
        labels.append("positive")
    elif score < -0.2:
        labels.append("negative")
    else:
        labels.append("neutral")

# Add the labels to the preprocessed corpus
labeled_corpus = []
for i in range(len(preprocessed_corpus)):
    labeled_corpus.append((preprocessed_corpus[i], labels[i]))

data = labeled_corpus
headers = ["text", "label"]
filename = "labeled_data.csv"

# Open the CSV file in write mode
with open(filename, mode="w", newline="", encoding="utf-8") as csvfile:
    # Create a CSV writer object
    writer = csv.writer(csvfile)
    # Write the headers to the CSV file
    writer.writerow(headers)
    # Write the data to the CSV file
    writer.writerows(data)
