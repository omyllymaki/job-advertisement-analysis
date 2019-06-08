import numpy as np
from matplotlib import pyplot as plt
from mpldatacursor import datacursor
from scipy.cluster import hierarchy as hc


def plot_dendogram(linkage_matrix, color_threshold=None):
    plt.figure(figsize=(16, 10))
    hc.dendrogram(
        linkage_matrix,
        truncate_mode='lastp',
        orientation='left',
        p=30,
        show_leaf_counts=True,
        leaf_rotation=0,
        leaf_font_size=8,
        show_contracted=True,
        color_threshold=color_threshold
    )
    plt.grid()


def plot_value_and_cumulative_value(values, value_label='Value', cumulative_value_label='Cumulative value'):
    fig, ax1 = plt.subplots()
    ax1.plot(values, 'b')
    ax1.set_ylabel(value_label, color='b')
    ax2 = ax1.twinx()
    ax2.plot(values.cumsum(), 'r')
    ax2.set_ylabel(cumulative_value_label, color='r')
    plt.grid()


def plot_mds(xs, ys, clusters, job_summary_list, legend_labels=None):
    plt.figure(figsize=(8, 8))
    colors = ['r', 'g', 'c', 'y', 'b', 'm']
    legend_elements = []
    clusters_unique = list(set(clusters))

    if not legend_labels:
        legend_labels = clusters_unique

    for k, color in zip(clusters_unique, colors):
        indices = np.where(clusters == k)[0]
        legend_label = legend_labels[k - 1]
        marker = color + 'o'

        job_ads = [job_summary_list[i] for i in indices]
        plot_x = xs[indices]
        plot_y = ys[indices]

        for x, y, ad in zip(plot_x, plot_y, job_ads):
            point = plt.plot(x, y, marker, label=ad)
            datacursor(point, formatter='{label}'.format, draggable=True)

        legend_element = plt.scatter([], [], color=marker[0], marker=marker[1], label=legend_label)
        legend_elements.append(legend_element)

    plt.legend(handles=legend_elements)


def plot_outlier_plot(deviations_scaled, threshold):
    plt.hist(deviations_scaled)
    plt.axvline(x=-threshold, color='r')
    plt.axvline(x=threshold, color='r')
    plt.grid()
