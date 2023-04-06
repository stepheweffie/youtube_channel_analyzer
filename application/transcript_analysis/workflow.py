import d6tflow
import openai
from youtube_transcript_api import YouTubeTranscriptApi
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

# Download NLTK resources


def nltk_downloads():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')


def download_transcript(video_id, language_code="en"):
    transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=[language_code])
    transcript_text = "\n".join([entry["text"] for entry in transcript_data])
    return transcript_text


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


class WordAnalysisTask(d6tflow.tasks.TaskPickle):
    processed_texts = {}

    def run(self):
        processed_text = self.input().load()
        # Perform a simple word frequency analysis
        word_frequencies = Counter()
        for video, processed_text in processed_text.items():
            word_frequencies.update(processed_text)

        # Get the most common words
        most_common_words = word_frequencies.most_common(10)
        print("Most common words:")
        for word, count in most_common_words:
            print(f"{word}: {count}")
        plot_word_histogram(most_common_words)


if __name__ == '__main__':
    nltk_downloads()