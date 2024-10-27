import pandas as pd
import os
from os import listdir
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

def analyze_sensor_data(data):
    sensor_columns = [col for col in data.columns if not col.startswith('MW')]
    importance_stats = {}
    for col in sensor_columns:
      importance_stats[col] = {
          'mean': data[col].mean(),
          'std': data[col].std(),
          'variance': data[col].var(),
          'range': data[col].max() - data[col].min()
      }
    return importance_stats

def calculate_score(stats):
    total_score = 0
    for col, stat_dict in stats.items():
        total_score += stat_dict['mean'] + stat_dict['std']
    return total_score

if __name__ == "__main__":
    dir = os.path.join(os.getcwd(), 'Rotem_model/data/new')
    data_stats = {}  # Dictionary to store data and importance stats

    for filename in os.listdir(dir):
        filepath = os.path.join(dir, filename)
        if not os.path.isfile(filepath) or not filename.endswith('.csv'):
            continue  # Skip non-CSV files

        df = pd.read_csv(filepath)
        columns_to_read = [f'S{i}' for i in range(1, 33)]
        X = df[columns_to_read].copy()
        importance_stats = analyze_sensor_data(X)
        score = calculate_score(importance_stats)

        data_stats[filename] = {
            'stats': importance_stats,
            'score': score
        }

    # Sort data_stats by score in descending order (highest to lowest)
    sorted_data_stats = dict(sorted(data_stats.items(), key=lambda x: x[1]['score'], reverse=True))
    top_files = list(sorted_data_stats.keys())


    top_files_data = []

    for filename in top_files:
        filepath = os.path.join(dir, filename)
        df = pd.read_csv(filepath)
        columns_to_read = [f'S{i}' for i in range(1, 33)]
        X = df[columns_to_read].copy()
        top_files_data.append(X)

    # Concatenate all dataframes into a single dataframe
    all_data = pd.concat(top_files_data, ignore_index=True)

    scaler = StandardScaler()
    all_data_scaled = scaler.fit_transform(all_data)

    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(all_data_scaled)

    cluster_centers = kmeans.cluster_centers_

    # Map each file to its cluster label
    file_clusters = {}
    for i, filename in enumerate(top_files):
        file_clusters[filename] = cluster_labels[i]

    # Find unique clusters and their representative files
    unique_clusters = set(cluster_labels)
    cluster_representatives = {}

    for cluster in unique_clusters:
        cluster_files = [filename for filename, label in file_clusters.items() if label == cluster]


        # Choose the representative file for each cluster (you can choose any heuristic here)
        if len(cluster_files)>0:
            representative_file = cluster_files  # For simplicity, choose the first file
            cluster_representatives[cluster] = representative_file

    # Reduce to 10 representative files
    representative_files = list(cluster_representatives.values())[:10]


    # all_data = pd.DataFrame()
    # for filename in top_files:
    #     filepath = os.path.join(dir, filename)
    #     df = pd.read_csv(filepath)
    #     all_data = pd.concat([all_data, df], ignore_index=True)
    #
    # # clean
    # missing_values = all_data.isnull().sum()
    #
    # # normalize
    # sensor_columns = [f'S{i}' for i in range(1, 33)]
    # scaler = StandardScaler()
    # all_data[sensor_columns] = scaler.fit_transform(all_data[sensor_columns])

    # # feature_selection
    # sensor_columns = [f'S{i}' for i in range(1, 33)]
    # target_columns = ['MSx', 'MSy', 'MSz', 'MEx', 'MEy', 'MEz', 'MWx', 'MWy', 'MWz']
    # X = all_data[sensor_columns]
    # y = all_data[target_columns]
