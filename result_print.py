from pprint import pprint

from matplotlib import pyplot as plt

from plotting import plot_dendogram


def print_clusters(clusters, job_summary_list, n_ads=5):
    clusters_unique = list(set(clusters))

    for k in clusters_unique:

        jobs = [i for j, i in zip(clusters, job_summary_list) if j == k]

        print(f'Cluster {k}')
        print(f'Job ads: {len(jobs)}')
        for job in jobs[:n_ads]:
            print(job)
        print('...')
        print('')


def display_clustering_results(linkage_matrix, clusters, job_summary_list, max_distance):
    plt.plot(linkage_matrix[-50:, 2], 'o')
    plot_dendogram(linkage_matrix, color_threshold=max_distance)
    print_clusters(clusters, job_summary_list, 5)


def display_most_similar_documents(document_index, top_indices, top_similarities, job_summary_list):
    target_document = job_summary_list[document_index]
    most_similar_documents = [(job_summary_list[i], s) for i, s in zip(top_indices, top_similarities)]
    print('Reference document: ', target_document)
    print('Most similar documents:')
    pprint(most_similar_documents)