from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calculate_most_similar_documents(document_index, similarity_matrix, n):
    indices = np.argsort(similarity_matrix[document_index])[::-1]
    top_indices = indices[:n]
    top_similarities = similarity_matrix[document_index, top_indices]
    return top_indices, top_similarities


def calculate_document_vector(tokens: List[str], model) -> np.ndarray:
    tokens = [token for token in tokens if token in model.wv.vocab]
    return np.mean(model[tokens], axis=0)


def search_documents_by_relevance(query_vector: np.ndarray,
                                  feature_array: np.ndarray,
                                  number_threshold: Optional[int] = None,
                                  similarity_threshold: Optional[float] = None) -> Tuple[List[int], List[float]]:
    similarity_array = cosine_similarity(query_vector, feature_array)
    indices = np.argsort(similarity_array)[0][::-1]
    if number_threshold:
        indices = indices[:number_threshold]
    similarities = similarity_array[0, indices]
    if similarity_threshold:
        indices = indices[similarities > similarity_threshold]
        similarities = similarities[similarities > similarity_threshold]
    return indices, similarities
