import numpy as np

from plotting import plot_outlier_plot


def outlier_detection(X, threshold, job_summary_list):
    outliers = detect_outliers(X, threshold)
    scaled_deviations = calculate_scaled_deviations(X)
    plot_outlier_plot(scaled_deviations, threshold)
    n_outliers = sum(outliers)
    outlier_proportion = round(100 * n_outliers / len(X), 2)

    print(f'Number of outliers detected: {n_outliers} ({outlier_proportion} %)')

    outlier_jobs = np.array(job_summary_list)[outliers]
    print('Outliers:')
    for o in outlier_jobs: print(o)


def detect_outliers(X, threshold):
    deviations_scaled = calculate_scaled_deviations(X)
    is_outlier_vector = abs(deviations_scaled) > threshold
    return is_outlier_vector


def calculate_scaled_deviations(X):
    median = np.median(X)
    deviations = X - median
    deviations_abs = np.abs(deviations)
    mad = np.median(deviations_abs)
    deviations_scaled = deviations / mad
    return deviations_scaled