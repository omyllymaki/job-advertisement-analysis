import numpy as np


def get_topic_key_words(terms, term_topic_matrix, n_terms=10):
    topics = []
    for component_vector in term_topic_matrix:
        indices_sorted = np.argsort(component_vector)
        top_indices = np.flip(indices_sorted[-n_terms:])
        component_terms = [terms[i] for i in top_indices]
        component_weights = component_vector[top_indices]
        topic = [(w, t) for w, t in zip(component_weights, component_terms)]
        topics.append(topic)
    return topics


def get_topics_of_documents(bag_array, n_topics=3):
    topics_list = []
    for row in bag_array:
        top_topic_indices = np.flip(np.argsort(row))[:n_topics]
        topics = [(row[i], i) for i in top_topic_indices]
        topics_list.append(topics)
    return topics_list


def topic_analysis(model, bag_array, terms, n_terms=10, n_topics=3):
    X = model.fit_transform(bag_array.copy())
    term_topic_matrix = model.components_
    topic_words = get_topic_key_words(terms, term_topic_matrix, n_terms)
    document_topics = get_topics_of_documents(X, n_topics)
    return topic_words, document_topics