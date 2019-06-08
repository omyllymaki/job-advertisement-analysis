import pickle
from gensim.models import fasttext


def save_pickle_file(data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def load_pickle_file(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def write_to_text_file(data, file_path):
    with open(file_path, 'w') as f:
        f.write(data)


def load_fasttext_model(file_path):
    return fasttext._load_fasttext_format(file_path)
