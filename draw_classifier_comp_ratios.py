import matplotlib.pyplot as plt
import numpy as np

# Data
names = ['DecisionTree', 'QDA', 'MLP', 'GaussianNB', 'LogisticRegression']

# Datasets with results for compression and clustering
datasets = {
    'orders': {
        'original': {'gzip': 3.587165199353363, 'lz4': 2.1650485446758676, 'zstd': 3.380769975922162},  # Original baseline
        'cluster': {'gzip': 3.7356990181186447, 'lz4': 2.2088561789664767, 'zstd': 3.558000921039849},  # Cluster baseline
        'gzip': [3.7364320899527663, 3.7361582189134315, 3.736517027615043, 3.736236838069756, 3.7344497250191595],
        'lz4': [2.2092670976020052, 2.2090028015165712, 2.209426404401593, 2.20910148641679, 2.206649289874437],
        'zstd': [3.55842430668807, 3.5581106945202072, 3.5587030534308113, 3.558183484459755, 3.555510854099642]
    },
    'partsupp': {
        'original': {'gzip': 4.164848836481552, 'lz4': 2.261516679794547, 'zstd': 3.655080624966541},
        'cluster': {'gzip': 4.3035069371402646, 'lz4': 2.3271561087224244, 'zstd': 3.7812391889114005},
        'gzip': [4.303000838102112, 4.303563665256417, 4.3031598625897205, 4.303538121748303, 4.29995062781266],
        'lz4': [2.326691595037761, 2.3270339491698997, 2.32691881201593, 2.327036240116266, 2.324992448692718],
        'zstd': [3.7808600810418644, 3.781224308643353, 3.7807582405106213, 3.7812618119823243, 3.779729982773125]
    },
    'flight': {
        'original': {'gzip': 7.2217097996194655, 'lz4': 4.270544560477076, 'zstd': 8.768682150562126},
        'cluster': {'gzip': 10.215361386188547, 'lz4': 5.923904474968684, 'zstd': 10.748068034572002},
        'gzip': [8.73893760594196, 10.133896145655397, 10.455245379571664, 9.99990429880647, 10.474770290020535],
        'lz4': [5.153406248094875, 5.8604062981148655, 6.023111878050532, 5.81309260640465, 6.06501630457628],
        'zstd': [9.896351741558487, 10.67817764163838, 10.788484644000027, 10.29792752496978, 10.867430487830546]
    },
    'nypd': {
        'original': {'gzip': 5.849014594521855, 'lz4': 3.2680472453182063, 'zstd': 6.092062471819926},
        'cluster': {'gzip': 6.37439887578076, 'lz4': 3.5605225152830853, 'zstd': 7.18715834600965},
        'gzip': [6.377318049933631, 6.406674088031725, 6.277530070518123, 6.367346423810145, 6.384605010438903],
        'lz4': [3.5586736172907365, 3.577840572525658, 3.5150512752759555, 3.5512302410895686, 3.5676287766549417],
        'zstd': [7.226507417663497, 7.246860703779565, 7.208774128562578, 7.111939196877623, 7.202652013368955]
    }
}

# Predefine colors for each compression method
comp_colors = {
    'gzip': 'red',
    'lz4': 'blue',
    'zstd': 'green'
}


def plot_dataset_with_baseline(dataset_name, comp_methods, names):
    # Setup figure
    plt.figure(figsize=(10, 6))

    # Extract original and cluster baselines
    original_baseline = comp_methods.pop('original')
    cluster_baseline = comp_methods.pop('cluster')

    # Plot the original baseline as a dashed line
    for method_name, baseline_value in original_baseline.items():
        plt.plot(names, [baseline_value] * len(names), linestyle='--', label=f'Original ({method_name})', color=comp_colors[method_name])

    # Plot the cluster baseline as a dashed line
    for method_name, baseline_value in cluster_baseline.items():
        plt.plot(names, [baseline_value] * len(names), linestyle=':', label=f'Cluster ({method_name})', color=comp_colors[method_name])

    # Plot compression ratios for each clustering configuration
    for method_name, comp_ratios in comp_methods.items():
        plt.plot(names, comp_ratios, marker='o', label=method_name, color=comp_colors[method_name])

    # Set chart title and labels
    plt.title(f'Compression Ratios for {dataset_name} with 10 Clusters and 0.1 Train size')
    plt.xticks(names, rotation=45, ha="right")
    plt.ylabel('Compression Ratio')
    plt.xlabel('Model Name')

    # Add a legend
    plt.legend()

    # Improve layout
    plt.tight_layout()
    plt.show()


# Loop over each dataset and plot it individually
for dataset_name, comp_methods in datasets.items():
    plot_dataset_with_baseline(dataset_name, comp_methods.copy(), names)
