import nltk
import string


# Preprocess the text data
def preprocess_text(text):
    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)

    # Remove punctuation and convert to lowercase
    preprocessed_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = sentence.translate(str.maketrans("", "", string.punctuation))
        preprocessed_sentences.append(sentence)

    # Join the preprocessed sentences back into a single text string
    preprocessed_text = " ".join(preprocessed_sentences)

    return preprocessed_text
