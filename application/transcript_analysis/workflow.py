
import d6tflow
# import os
import openai
from main import get_channel_id, download_transcript, get_video_ids
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from collections import Counter

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def plot_word_histogram(most_common_words):
    words, counts = zip(*most_common_words)

    plt.figure(figsize=(10, 5))
    plt.bar(words, counts)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Word Frequency Histogram')
    plt.show()


def chat_gpt_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )

    return response.choices[0].text.strip()


def process_video_transcripts(channel_url):
    channel_id = get_channel_id(channel_url)
    video_ids = get_video_ids(channel_id)
    processed_texts = {}

    for video_id in video_ids:
        nlp_task = NLPAnalysisTask(video_id=video_id)
        d6tflow.run(nlp_task)

        processed_text = nlp_task.output().load()
        print(f"Processed text for video ID {video_id}")

        # Perform a simple word frequency analysis
        word_frequencies = Counter()
        for video_id, processed_text in processed_texts.items():
            word_frequencies.update(processed_text)

        # Get the most common words
        most_common_words = word_frequencies.most_common(10)
        print("Most common words:")
        for word, count in most_common_words:
            print(f"{word}: {count}")
        # Plot the histogram
        plot_word_histogram(most_common_words)


# Preprocessing
def preprocess(text):
    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Lowercasing
    tokens = [token.lower() for token in tokens]

    # Removing punctuation and special characters
    tokens = [token for token in tokens if token.isalnum()]

    # Removing stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens


class DownloadTranscriptTask(d6tflow.tasks.TaskPickle):
    video_id = d6tflow.Parameter()
    language_code = d6tflow.Parameter(default="en")

    def run(self):
        transcript_text = download_transcript(self.video_id, self.language_code)
        if transcript_text:
            self.save(transcript_text)


class NLPAnalysisTask(d6tflow.tasks.TaskPickle):
    video_id = d6tflow.Parameter()

    def requires(self):
        return DownloadTranscriptTask(video_id=self.video_id)

    def run(self):
        transcript_text = self.input().load()
        processed_text = preprocess(transcript_text)
        # Perform any additional NLP analysis here
        self.save(processed_text)


class ChatGPTTask(d6tflow.tasks.TaskPickle):
    video_id = d6tflow.Parameter()

    def requires(self):
        return NLPAnalysisTask(video_id=self.video_id)

    def run(self):
        processed_text = self.input().load()
        transcript_summary = " ".join(processed_text[:100])  # Adjust the summary length as needed
        prompt = f"The transcript summary is: {transcript_summary}. What is your response?"
        gpt_response = chat_gpt_response(prompt)
        self.save(gpt_response)