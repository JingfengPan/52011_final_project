import matplotlib.pyplot as plt

# Data
xlabels = ['4 clus', '6 clus', '8 clus', '10 clus', '12 clus']

# Datasets with results for compression and clustering
datasets = {
    'orders': {
        'original': {'gzip': 3.587165199353363, 'lz4': 2.1650485446758676, 'zstd': 3.380769975922162},  # Baseline values
        'gzip': [3.7465870332673368, 3.7434081822464336, 3.7366244952634036, 3.7364899978532717, 3.7344497250191595],
        'lz4': [2.2229673448215665, 2.2169898557709327, 2.2111001841007814, 2.208966043859751, 2.206649289874437],
        'zstd': [3.5754042032879627, 3.5664267806904446, 3.560294052957705, 3.5585046870622383, 3.555510854099642]
    },
    'partsupp': {
        'original': {'gzip': 4.164848836481552, 'lz4': 2.261516679794547, 'zstd': 3.655080624966541},
        'gzip': [4.315984323706778, 4.3131453316277035, 4.30741071674585, 4.303519473611786, 4.29995062781266],
        'lz4': [2.334125024662589, 2.332571339790226, 2.3293348636939997, 2.32696778858506, 2.324992448692718],
        'zstd': [3.7867306132844587, 3.7870086005124084, 3.7832085304211, 3.781276571564887, 3.779729982773125]
    },
    'flight': {
        'original': {'gzip': 7.2217097996194655, 'lz4': 4.270544560477076, 'zstd': 8.768682150562126},
        'gzip': [9.276475830147577, 9.717886967232399, 10.08440517070856, 10.221245925158781, 10.474770290020535],
        'lz4': [5.4156433039903895, 5.635623132036781, 5.82648082143404, 5.926002243542894, 6.06501630457628],
        'zstd': [10.095217063759065, 10.377594205634475, 10.56209228977164, 10.74260224188829, 10.867430487830546]
    },
    'nypd': {
        'original': {'gzip': 5.849014594521855, 'lz4': 3.2680472453182063, 'zstd': 6.092062471819926},
        'gzip': [6.306931845610923, 6.408725064218285, 6.424805624054053, 6.390189049399048, 6.384605010438903],
        'lz4': [3.5237225079551995, 3.5787393618215706, 3.5874310045173603, 3.5695817479869705, 3.5676287766549417],
        'zstd': [7.256531939171327, 7.36741731125765, 7.276737578464417, 7.209674270358301, 7.202652013368955]
    }
}

# Predefine colors for each compression method
comp_colors = {
    'gzip': 'red',
    'lz4': 'blue',
    'zstd': 'green'
}


def plot_dataset_with_baseline(dataset_name, comp_methods, xlabels):
    # Setup figure
    plt.figure(figsize=(10, 6))

    # Extract original (baseline) compression ratios for each method
    baseline = comp_methods.pop('original')

    # Plot the baseline values as dashed lines for each method with the same color
    for method_name, baseline_value in baseline.items():
        plt.plot(xlabels, [baseline_value] * len(xlabels), linestyle='--', label=f'Original ({method_name})', color=comp_colors[method_name])

    # Plot compression ratios for each clustering configuration using the same color
    for method_name, comp_ratios in comp_methods.items():
        plt.plot(xlabels, comp_ratios, marker='o', label=method_name, color=comp_colors[method_name])

    # Set chart title and labels
    plt.title(f'Compression Ratios for {dataset_name}')
    plt.xticks(xlabels, rotation=45, ha="right")
    plt.ylabel('Compression Ratio')
    plt.xlabel('Number of Clusters')

    # Add a legend
    plt.legend()

    # Improve layout
    plt.tight_layout()
    plt.show()


# Loop over each dataset and plot it individually
for dataset_name, comp_methods in datasets.items():
    plot_dataset_with_baseline(dataset_name, comp_methods.copy(), xlabels)
