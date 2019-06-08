import re
from typing import Callable, List

from langdetect import detect_langs
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords

from voikko_lemmatizer import VoikkoLemmatizer

STOPWORDS = stopwords.words('english') + stopwords.words('finnish')


def tokenize_and_clean_text(text: str,
                            tokenization_method: Callable,
                            lemmatization_method: Callable = None,
                            strip_character_list: List[str] = None,
                            stopwords: List[str] = None):
    text = text.lower()
    if strip_character_list:
        text = remove_characters(text, strip_character_list)

    tokens = tokenization_method(text)
    if stopwords:
        tokens = remove_stopwods(stopwords, tokens)

    if lemmatization_method:
        tokens = lemmatization_method(tokens)

    tokens = remove_non_alphabetic_characters(tokens)
    return tokens


def remove_non_alphabetic_characters(tokens):
    tokens = [token for token in tokens if token.isalpha()]
    return tokens


def remove_stopwods(stopwords, tokens):
    tokens = [token for token in tokens if token not in stopwords]
    return tokens


def lemmatize_english_tokens(tokens):
    lemmatization_method = WordNetLemmatizer().lemmatize
    lemmas = [lemmatization_method(token) for token in tokens]
    return lemmas


def lemmatize_finnish_tokens(tokens):
    return VoikkoLemmatizer().lemmatize(tokens)


def remove_characters(text, characters):
    for c in characters:
        text = text.replace(c, '')
    return text


def tokenize_and_clean_finnish_text(text: str):
    tokenization_method = re.compile(r'\S+').findall
    strip_character_list = ['.', ',', '-']
    lemmatization_method = lemmatize_finnish_tokens
    tokens = tokenize_and_clean_text(text,
                                     tokenization_method=tokenization_method,
                                     strip_character_list=strip_character_list,
                                     stopwords=STOPWORDS,
                                     lemmatization_method=lemmatization_method)
    return tokens


def tokenize_and_clean_english_text(text: str):
    tokenization_method = word_tokenize
    strip_character_list = ['.', ',', '-']
    lemmatization_method = lemmatize_english_tokens
    tokens = tokenize_and_clean_text(text,
                                     tokenization_method=tokenization_method,
                                     strip_character_list=strip_character_list,
                                     stopwords=STOPWORDS,
                                     lemmatization_method=lemmatization_method)
    return tokens


def detect_language(text):
    results = detect_langs(text)
    results_dict = [result.__dict__ for result in results]
    language = results_dict[0].get('lang', None)
    probability = results_dict[0].get('prob', None)
    return language, probability


def process_text(text: str):
    language, probability = detect_language(text)
    processing_methods_by_language = {
        'en': tokenize_and_clean_english_text,
        'fi': tokenize_and_clean_finnish_text,
    }
    processing_method = processing_methods_by_language.get(language, re.compile(r'\S+').findall)
    tokens = processing_method(text)
    return tokens


def process_documents(documents: List[str]):
    return [process_text(document) for document in documents]
