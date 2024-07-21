import os
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import re
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# def load_documents(path):
#     documents = []
#     for filename in os.listdir(path):
#         with open(os.path.join(path, filename), "r", encoding="utf-8") as file:
#             documents.append(file.read())
#     return documents

# def preprocess_documents(documents):
#     preprocessed_documents = []
#     stop_words = set(stopwords.words("english"))

#     for doc in documents:
#         # Convert to lowercase
#         doc = doc.lower()

#         # Remove punctuation
#         doc = doc.translate(str.maketrans("", "", string.punctuation))

#         # Tokenize
#         tokens = word_tokenize(doc)

#         # Remove stopwords
#         tokens = [token for token in tokens if token not in stop_words]

#         preprocessed_documents.append(tokens)

#     return preprocessed_documents



# function to load the documents with labels for evaluation
def load_documents_and_labels(path):
    documents = []
    labels_true = []
    for label, class_folder in enumerate(os.listdir(path)):
        class_path = os.path.join(path, class_folder)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                with open(os.path.join(class_path, filename), "r", encoding="utf-8") as file:
                    documents.append(file.read())
                    labels_true.append(label)

    return documents, labels_true


# function to preprocess each document
def preprocess_documents(documents):
    preprocessed_documents = []
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    for doc in documents:
        # Remove HTML tags
        doc = BeautifulSoup(doc, "html.parser").get_text()
        # Remove URLs
        doc = re.sub(r'http\S+|www\S+|https\S+', '', doc, flags=re.MULTILINE)
        # Remove email addresses
        doc = re.sub(r'\S+@\S+', '', doc, flags=re.MULTILINE)
        # Convert to lowercase
        doc = doc.lower()
        # Remove punctuation
        doc = doc.translate(str.maketrans("", "", string.punctuation))
        # Tokenize
        tokens = word_tokenize(doc)
        # Remove stopwords and lemmatize
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        # Remove extra spaces
        tokens = [token.strip() for token in tokens]
        preprocessed_documents.append(tokens)

    return preprocessed_documents