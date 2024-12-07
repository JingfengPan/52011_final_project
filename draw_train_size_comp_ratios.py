import matplotlib.pyplot as plt
import numpy as np

# Data
train_sizes = [0.05, 0.1, 0.15, 0.2, 0.25]

# Datasets with results for compression and clustering
datasets = {
    'orders': {
        'original': {'gzip': 3.587165199353363, 'lz4': 2.1650485446758676, 'zstd': 3.380769975922162},  # Original baseline
        'cluster': {'gzip': 3.7356990181186447, 'lz4': 2.2088561789664767, 'zstd': 3.558000921039849},  # Cluster baseline
        'gzip': [3.73639195681737, 3.7364320899527663, 3.7363625537000775, 3.736376559026921, 3.736374511457964],
        'lz4': [2.209239465379382, 2.2092670976020052, 2.2092967922739337, 2.20933917370907, 2.2093367395961487],
        'zstd': [3.5584156894143097, 3.55842430668807, 3.558570063796659, 3.558395929271718, 3.558295571935268]
    },
    'partsupp': {
        'original': {'gzip': 4.164848836481552, 'lz4': 2.261516679794547, 'zstd': 3.655080624966541},
        'cluster': {'gzip': 4.3035069371402646, 'lz4': 2.3271561087224244, 'zstd': 3.7812391889114005},
        'gzip': [4.303036245493011, 4.303000838102112, 4.3030674232798285, 4.303100324951852, 4.302983291292471],
        'lz4': [2.3268076259056856, 2.326691595037761, 2.3267400123545894, 2.3267421653000495, 2.3268014873858833],
        'zstd': [3.7807465086403225, 3.7808600810418644, 3.7808978190689353, 3.780833471347453, 3.780837583730301]
    },
    'flight': {
        'original': {'gzip': 7.2217097996194655, 'lz4': 4.270544560477076, 'zstd': 8.768682150562126},
        'cluster': {'gzip': 10.215361386188547, 'lz4': 5.923904474968684, 'zstd': 10.748068034572002},
        'gzip': [9.015614867506791, 8.73893760594196, 8.707332086785273, 8.979866691902787, 8.913410630913786],
        'lz4': [5.300274487295112, 5.153406248094875, 5.128710599576134, 5.274207024936287, 5.250190439152036],
        'zstd': [10.076428560631731, 9.896351741558487, 9.882406985016157, 10.07884683730187, 10.036208045902097]
    },
    'nypd': {
        'original': {'gzip': 5.849014594521855, 'lz4': 3.2680472453182063, 'zstd': 6.092062471819926},
        'cluster': {'gzip': 6.37439887578076, 'lz4': 3.5605225152830853, 'zstd': 7.18715834600965},
        'gzip': [6.403862909843192, 6.377318049933631, 6.422306921342175, 6.422579357600946, 6.422495606216443],
        'lz4': [3.5719079132873683, 3.5586736172907365, 3.5844337111234728, 3.5851568997131493, 3.5839654681748945],
        'zstd': [7.248165547660122, 7.226507417663497, 7.316870773625634, 7.318565183099217, 7.320369179856991]
    }
}

# Predefine colors for each compression method
comp_colors = {
    'gzip': 'red',
    'lz4': 'blue',
    'zstd': 'green'
}


def plot_dataset_with_baseline(dataset_name, comp_methods, train_sizes):
    # Setup figure
    plt.figure(figsize=(10, 6))

    # Extract original and cluster baselines
    original_baseline = comp_methods.pop('original')
    cluster_baseline = comp_methods.pop('cluster')

    # Plot the original baseline as a dashed line
    for method_name, baseline_value in original_baseline.items():
        plt.plot(train_sizes, [baseline_value] * len(train_sizes), linestyle='--', label=f'Original ({method_name})', color=comp_colors[method_name])

    # Plot the cluster baseline as a dashed line
    for method_name, baseline_value in cluster_baseline.items():
        plt.plot(train_sizes, [baseline_value] * len(train_sizes), linestyle=':', label=f'Cluster ({method_name})', color=comp_colors[method_name])

    # Plot compression ratios for each train size
    for method_name, comp_ratios in comp_methods.items():
        plt.plot(train_sizes, comp_ratios, marker='o', label=method_name, color=comp_colors[method_name])

    # Set chart title and labels
    plt.title(f'Compression Ratios for {dataset_name} with 10 Clusters and QDA Classifier')
    plt.xticks(train_sizes)
    plt.ylabel('Compression Ratio')
    plt.xlabel('Train Size')

    # Add a legend
    plt.legend()

    # Improve layout
    plt.tight_layout()
    plt.show()


# Loop over each dataset and plot it individually
for dataset_name, comp_methods in datasets.items():
    plot_dataset_with_baseline(dataset_name, comp_methods.copy(), train_sizes)
