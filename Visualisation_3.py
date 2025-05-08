# load data RNAseq
# filter for JA, SA, SAJA treatments
# first plot MYC2 for JA, SA, SAJA treatments
# for each tf, determine if it is activated, not regulated, or repressed
# plot 3x3 drid of timeseries for each tf
# First line JA, second line SA, third line SAJA
# first column activated, second column not regulated, third column repressed

# Calculate cumulative expression during treatment
# Plot the cumulative expression for each TF in each treatment

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.ticker as ticker
from matplotlib.colors import SymLogNorm
from matplotlib import rcParams
from matplotlib.cm import get_cmap

# import data
# Import data with MultiIndex
df_JA = pd.read_csv(
    "C:/Users/sergej.pracek/OneDrive - NIB/Desktop/SKM/DAP-seq+RNAseq/RNA_seq_time_series_urejeno_JA.csv",
    index_col=0,  # Use the first column as the index (gene names)
    header=[0, 1, 2]  # Use the first three rows as the MultiIndex for columns
)
df_SA = pd.read_csv(
    "C:/Users/sergej.pracek/OneDrive - NIB/Desktop/SKM/DAP-seq+RNAseq/RNA_seq_time_series_urejeno_SA.csv",
    index_col=0,  # Use the first column as the index (gene names)
    header=[0, 1, 2]  # Use the first three rows as the MultiIndex for columns
)
df_SAJA = pd.read_csv(
    "C:/Users/sergej.pracek/OneDrive - NIB/Desktop/SKM/DAP-seq+RNAseq/RNA_seq_time_series_urejeno_SAJA.csv",
    index_col=0,  # Use the first column as the index (gene names)
    header=[0, 1, 2]  # Use the first three rows as the MultiIndex for columns
)

# Load the data from the TSV file
DAPseq_path = "C:/Users/sergej.pracek/OneDrive - NIB/Desktop/SKM/DAP-seq+RNAseq/dapseq-MYC2-tfs - 5utr.tsv"
df_DAPseq = pd.read_csv(DAPseq_path, sep="\t")

# Create a subset of RNA-seq datasets with only genes in df_DAPseq['tf']
dapseq_genes = pd.concat([df_DAPseq['tf'], df_DAPseq['target']]).unique()   # Combine unique TFs and target - this way MYC2 is also included in analysis

# Filter the RNA-seq datasets
df_JA_subset = df_JA[df_JA.index.isin(dapseq_genes)]
df_SA_subset = df_SA[df_SA.index.isin(dapseq_genes)]
df_SAJA_subset = df_SAJA[df_SAJA.index.isin(dapseq_genes)]

# Verify the subsets
print(f"Subset of df_JA: {df_JA_subset.shape}")
print(f"Subset of df_SA: {df_SA_subset.shape}")
print(f"Subset of df_SAJA: {df_SAJA_subset.shape}")

def reorder_by_timepoint(df):
    """
    Reorder the columns of a DataFrame based on timepoints from smallest to largest.
    
    Parameters:
    - df: DataFrame with MultiIndex columns where the second level represents timepoints.
    
    Returns:
    - Reordered DataFrame.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Extract timepoints from the second level of the MultiIndex
        timepoints = df.columns.get_level_values(1)
        # Sort columns based on numeric timepoints
        sorted_columns = sorted(df.columns, key=lambda col: float(col[1]))
        return df[sorted_columns]
    else:
        raise ValueError("The DataFrame does not have MultiIndex columns.")

# Reorder the subsets
df_JA_subset = reorder_by_timepoint(df_JA_subset)
df_SA_subset = reorder_by_timepoint(df_SA_subset)
df_SAJA_subset = reorder_by_timepoint(df_SAJA_subset)

# Verify the reordering
print(f"Reordered df_JA_subset columns: {df_JA_subset.columns}")
print(f"Reordered df_SA_subset columns: {df_SA_subset.columns}")
print(f"Reordered df_SAJA_subset columns: {df_SAJA_subset.columns}")

def calculate_integral_between_curves(df, treatment_label):
    """
    Calculate the integral (area under the curve) between treatment and Mock for all genes.
    For each timepoint, calculate the average for Mock and treatment, then subtract Mock from treatment.
    
    Parameters:
    - df: DataFrame containing expression data with time points as columns.
           Assumes that 'Mock' columns are present in the DataFrame.
    - treatment_label: Name of the treatment column.
    
    Returns:
    - DataFrame with an additional column for the integral values.
    """
    # Separate treatment and Mock columns
    treatment_cols = df.loc[:, df.columns.get_level_values('treatment') == treatment_label]
    mock_cols = df.loc[:, df.columns.get_level_values('treatment') == 'Mock']
    
    # Ensure the treatment and Mock columns align correctly
    if treatment_cols.shape[1] != mock_cols.shape[1]:
        raise ValueError("Mismatch between treatment and Mock column shapes.")
    
    # Calculate the average for each timepoint
    treatment_avg = treatment_cols.T.groupby(level="timepoint").mean().T
    mock_avg = mock_cols.T.groupby(level="timepoint").mean().T
    
    # Subtract Mock averages from treatment averages
    corrected_df = treatment_avg - mock_avg
    
    # Extract timepoints from column names (second level of MultiIndex)
    timepoints = [float(tp) for tp in corrected_df.columns]
    
    # Sort timepoints and corresponding columns
    sorted_indices = np.argsort(timepoints)
    timepoints = np.array(timepoints)[sorted_indices]
    corrected_df = corrected_df.iloc[:, sorted_indices]
    
    # Calculate the integral (area under the curve) for each gene
    integral_values = np.trapz(corrected_df.values, x=timepoints, axis=1)
    
    # Add the integral values as a new column
    df[f"{treatment_label}_vs_Mock_integral"] = integral_values
    
    return df

# Apply the function to each dataset
df_JA_with_integral = calculate_integral_between_curves(df_JA_subset, "JA")
df_SA_with_integral = calculate_integral_between_curves(df_SA_subset, "SA")
df_SAJA_with_integral = calculate_integral_between_curves(df_SAJA_subset, "SAJA")

# Display a sample of the updated DataFrame
print(df_JA_with_integral[[f"JA_vs_Mock_integral"]].head())
print(df_SA_with_integral[[f"SA_vs_Mock_integral"]].head())
print(df_SAJA_with_integral[[f"SAJA_vs_Mock_integral"]].head())

# To determine linear treshold for the symmetric log transformation, we can use the absolute minimum value
# from the integral values. This will help us to set the linthresh parameter for the symlog transformation.
# A rule of thumb: lintresh cca.=  min |X|
def get_top_genes_with_min_absolute_value(datasets, top_n=5):
    """
    Find the top N genes with the lowest absolute value for the treatment_vs_Mock_integral column in each dataset.
    
    Parameters:
    - datasets: List of tuples [(df, treatment_label)] where df is the DataFrame and
                treatment_label is the name of the treatment.
    - top_n: Number of genes to return with the lowest absolute values.
    
    Returns:
    - Dictionary with the top N genes and their absolute minimum values for each dataset.
    """
    top_genes = {}
    for df, treatment_label in datasets:
        integral_col = f"{treatment_label}_vs_Mock_integral"
        if integral_col in df.columns:
            # Get the top N genes with the lowest absolute values
            abs_min_genes = df[integral_col].abs().nsmallest(top_n).index
            abs_min_values = df[integral_col].loc[abs_min_genes]
            top_genes[treatment_label] = abs_min_values.to_dict()
        else:
            print(f"Warning: {integral_col} not found in the dataset for {treatment_label}.")
    return top_genes

# Example usage
datasets = [
    (df_JA_with_integral, "JA"),
    (df_SA_with_integral, "SA"),
    (df_SAJA_with_integral, "SAJA")
]

top_genes = get_top_genes_with_min_absolute_value(datasets, top_n=5)

# Print the results
for label, genes in top_genes.items():
    print(f"{label} - Top 5 Genes with Lowest Absolute Min Values:")
    for gene, value in genes.items():
        print(f"  Gene: {gene}, Value: {value}")

# Just to check: MYC expression in JA, SA, SAJA
print("MYC:" + str({label: df.loc["AT1G32640", f"{label}_vs_Mock_integral"] for df, label in [(df_JA_with_integral, "JA"), (df_SA_with_integral, "SA"), (df_SAJA_with_integral, "SAJA")]}))

# As some genes have cumulative expression = 0
# it is not possible to plot them on a log scale
# I will use symlog scale with linthresh = 0.1 for plotting

# creation of table with all genes and their integral values
def create_integral_table(df_JA, df_SA, df_SAJA, df_DAPseq):
    """
    Create a new dataset containing treatment_vs_mock_integral values for each treatment (JA, SA, SAJA) for each gene.
    Also adds the peak_signalValue column from df_DAPseq based on the match in tf and Gene.

    Parameters:
    - df_JA: DataFrame containing the JA_vs_Mock_integral column.
    - df_SA: DataFrame containing the SA_vs_Mock_integral column.
    - df_SAJA: DataFrame containing the SAJA_vs_Mock_integral column.
    - df_DAPseq: DataFrame containing the tf and peak_signalValue columns.

    Returns:
    - A new DataFrame with columns: ['Gene', 'JA_integral', 'SA_integral', 'SAJA_integral', 'peak_signalValue'].
    """
    # Extract the integral columns
    ja_integral = df_JA["JA_vs_Mock_integral"].rename("JA_integral")
    sa_integral = df_SA["SA_vs_Mock_integral"].rename("SA_integral")
    saja_integral = df_SAJA["SAJA_vs_Mock_integral"].rename("SAJA_integral")

    # Combine the integral columns into a single DataFrame
    combined_integral = pd.concat([ja_integral, sa_integral, saja_integral], axis=1)

    # Reset the index to include the gene names as a column
    combined_integral.reset_index(inplace=True)
    combined_integral.rename(columns={"index": "gene"}, inplace=True)

    # Handle duplicates in df_DAPseq by keeping the largest peak_signalValue for each tf
    #df_DAPseq_max = df_DAPseq.groupby("tf", as_index=False)["peak_signalValue"].max()

    # Map peak_signalValue to the combined_integral DataFrame based on Gene
    #combined_integral["peak_signalValue"] = combined_integral["gene"].map(
    #    df_DAPseq_max.set_index("tf")["peak_signalValue"]
    #)

    return combined_integral

# Create the new dataset
integral_table = create_integral_table(df_JA_with_integral, df_SA_with_integral, df_SAJA_with_integral, df_DAPseq)

# Display the first few rows of the new dataset
print(integral_table.head())
# MYC check-up - NaN values in peak_signalValue column is expected, as MYC is not in the DAPseq data
print(integral_table[integral_table.iloc[:, 0] == "AT1G32640"])

# plotting the integral values
def plot_final_integral_expression_comparison(integral_table, title, highlight_gene="AT1G32640"):
    """
    Plot the integral expression comparison between JA and SA datasets.
    Colors are based on the logarithmic values of SAJA_integral,
    and highlight a specific selected gene.

    Parameters:
    - integral_table: DataFrame containing columns ['gene', 'JA_integral', 'SA_integral', 'SAJA_integral', 
        'peak_signalValue'].
    - title: Title of the plot.
    - highlight_gene: Gene to highlight specifically (default: "AT1G32640").
    """
    # Ensure the required columns exist
    required_columns = ['gene', 'JA_integral', 'SA_integral', 'SAJA_integral']
    for col in required_columns:
        if col not in integral_table.columns:
            raise ValueError(f"Column '{col}' not found in the integral_table.")

    # Apply logarithmic transformation to SAJA_integral for coloring
    log_saja_integral = np.log1p(np.abs(integral_table['SAJA_integral'])) * np.sign(integral_table['SAJA_integral'])

    # Plot
    plt.figure(figsize=(8, 8))
    plt.xscale('symlog', linthresh=50)
    plt.yscale('symlog', linthresh=50)

    # Scatter plot for all genes
    scatter = plt.scatter(
        integral_table['JA_integral'],
        integral_table['SA_integral'],
        c=log_saja_integral,  # Use the logarithmic values directly for coloring
        cmap='RdBu',       # Use a colormap
        alpha=0.7,
        edgecolor='k',
        label="Genes"
    )

    # Highlight the specific gene if it exists
    if highlight_gene in integral_table['gene'].values:
        highlight_row = integral_table[integral_table['gene'] == highlight_gene]
        plt.scatter(
            highlight_row['JA_integral'],
            highlight_row['SA_integral'],
            c='none',  # No fill color
            edgecolor="red",  # Red border
            linewidth=2,  # Thickness of the border
            s=100,  # Size of the marker
            label=f"Highlighted Gene: {highlight_gene}"
        )
    else:
        print(f"Warning: {highlight_gene} not found in the dataset. Skipping highlight.")

    # Add reference lines
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=plt.gca(), pad=0.1)  # Explicitly associate the colorbar with the current Axes
    cbar.set_label("Logarithmic SAJA Integral Value")

    # Customize plot
    plt.title(title)
    plt.xlabel("Integral Expression (JA vs Mock)")
    plt.ylabel("Integral Expression (SA vs Mock)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Apply the function to the JA and SA datasets
plot_final_integral_expression_comparison(
    integral_table,
    "Integral Expression: JA vs SA (Top 20 Genes Highlighted)",
    highlight_gene="AT1G32640"
)

# there is a formation of groups of genes in the plot

def cluster_and_separate_datasets(integral_table, n_clusters=6, highlight_gene="AT1G32640"):
    """
    Cluster genes based on logarithmic values of integral expressions, plot the clusters,
    and separate the genes into datasets based on their clusters.

    Parameters:
    - integral_table: DataFrame containing columns ['gene', 'JA_integral', 'SA_integral', 'SAJA_integral'].
    - n_clusters: Number of clusters for K-Means.
    - highlight_gene: Gene to highlight specifically (default: "AT1G32640").

    Returns:
    - A dictionary where keys are cluster numbers and values are DataFrames for each cluster.
    """
    # Ensure the required columns exist
    required_columns = ['JA_integral', 'SA_integral', 'SAJA_integral']
    for col in required_columns:
        if col not in integral_table.columns:
            raise ValueError(f"Column '{col}' not found in the integral_table.")

    # Apply logarithmic transformation
    log_integrals = np.log1p(np.abs(integral_table[required_columns])) * np.sign(integral_table[required_columns])

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(log_integrals)

    # Add cluster labels to the table
    integral_table['Cluster'] = clusters

    # Separate genes into datasets based on clusters
    cluster_datasets = {}
    for cluster in range(n_clusters):
        cluster_datasets[cluster] = integral_table[integral_table['Cluster'] == cluster]

    # Plot the clusters
    plt.figure(figsize=(8, 8))
    plt.xscale('symlog', linthresh=50)
    plt.yscale('symlog', linthresh=50)

    # Scatter plot for each cluster
    for cluster in range(n_clusters):
        cluster_data = cluster_datasets[cluster]
        plt.scatter(
            cluster_data['JA_integral'],
            cluster_data['SA_integral'],
            label=f"Cluster {cluster}",
            alpha=0.7
        )

    # Highlight the specific gene if it exists
    if highlight_gene in integral_table['gene'].values:
        highlight_row = integral_table[integral_table['gene'] == highlight_gene]
        plt.scatter(
            highlight_row['JA_integral'],
            highlight_row['SA_integral'],
            c='none',
            edgecolor="red",
            linewidth=1.5,
            s=50,
            label=f"Highlighted Gene: {highlight_gene}"
        )
    else:
        print(f"Warning: {highlight_gene} not found in the dataset. Skipping highlight.")

    # Add reference lines
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)

    # Customize plot
    plt.title("Clusters of Genes Based on Logarithmic Integral Values")
    plt.xlabel("Integral Expression (JA vs Mock)")
    plt.ylabel("Integral Expression (SA vs Mock)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return cluster_datasets

# Apply the clustering and separate datasets function
clustered_datasets = cluster_and_separate_datasets(integral_table, n_clusters=6, highlight_gene="AT1G32640")

# Access individual datasets
cluster_0 = clustered_datasets[0]
cluster_1 = clustered_datasets[1]
cluster_2 = clustered_datasets[2]
cluster_3 = clustered_datasets[3]
cluster_4 = clustered_datasets[4]
cluster_5 = clustered_datasets[5]

def plot_histograms_for_clusters_subplots(clustered_datasets):
    """
    Plot histograms of peak_signalValue for each cluster as subplots with the same x-axis and y-axis.

    Parameters:
    - clustered_datasets: Dictionary where keys are cluster numbers and values are DataFrames for each cluster.
    """
    n_clusters = len(clustered_datasets)
    fig, axes = plt.subplots(n_clusters, 1, figsize=(8, 4 * n_clusters), sharex=True, sharey=True)  # Share x and y axes

    # Ensure axes is iterable (even if there's only one cluster)
    if n_clusters == 1:
        axes = [axes]

    # Determine the global x-axis range
    all_peak_signal_values = pd.concat([clustered_datasets[cluster]['peak_signalValue'].dropna() for cluster in clustered_datasets])
    x_min, x_max = all_peak_signal_values.min(), all_peak_signal_values.max()

    for cluster, ax in zip(clustered_datasets.keys(), axes):
        # Drop NaN values in peak_signalValue to avoid errors in the histogram
        peak_signal_values = clustered_datasets[cluster]['peak_signalValue'].dropna()

        # Plot histogram
        ax.hist(peak_signal_values, bins=20, color='blue', alpha=0.7, edgecolor='black')
        ax.set_title(f"Cluster {cluster}")
        ax.set_xlim(x_min, x_max)  # Set the same x-axis range for all subplots
        ax.set_xlabel("Peak Signal Value")
        ax.set_ylabel("Frequency")
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()
    plt.show()

# Call the function to plot histograms for each cluster as subplots
plot_histograms_for_clusters_subplots(clustered_datasets)

def filter_and_group_genes(integral_table):
    """
    Filter and group genes into Activators and Repressors based on integral values.

    Activators:
    - JA_integral > 50
    - SA_integral < -50
    - SAJA_integral > 50

    Repressors:
    - JA_integral < 50
    - SA_integral > 50
    - SAJA_integral < -50

    Parameters:
    - integral_table: DataFrame containing columns ['JA_integral', 'SA_integral', 'SAJA_integral'].

    Returns:
    - A dictionary with two keys: 'Activators' and 'Repressors', each containing the corresponding DataFrame.
    """
    # Filter for Activators
    activators = integral_table[
        (integral_table['JA_integral'] > 50) &
        (integral_table['SA_integral'] < -50) &
        (integral_table['SAJA_integral'] > 50)
    ]

    # Filter for Repressors
    repressors = integral_table[
        (integral_table['JA_integral'] < 50) &
        (integral_table['SA_integral'] > 50) &
        (integral_table['SAJA_integral'] < -50)
    ]

    # Return the grouped DataFrames
    return {
        "Activators": activators,
        "Repressors": repressors
    }

# Apply the function to the integral_table
grouped_genes = filter_and_group_genes(integral_table)

# Access the Activators and Repressors
activators = grouped_genes["Activators"]
repressors = grouped_genes["Repressors"]

# Display the results
print("Activators:")
print(activators)

print("Repressors:")
print(repressors)

# Set the font to Roboto
rcParams['font.family'] = 'Arial'

def plot_final_integral_expression_comparison_with_circles(integral_table, title, highlight_gene="AT1G32640"):
    """
    Plot the integral expression comparison between JA and SA datasets.
    Colors are based on the logarithmic values of SAJA_integral,
    and highlight a specific selected gene. Each activator and repressor is circled individually.

    Parameters:
    - integral_table: DataFrame containing columns ['gene', 'JA_integral', 'SA_integral', 'SAJA_integral', 
        'peak_signalValue'].
    - title: Title of the plot.
    - highlight_gene: Gene to highlight specifically (default: "AT1G32640").
    """
    # Ensure the required columns exist
    required_columns = ['gene', 'JA_integral', 'SA_integral', 'SAJA_integral']
    for col in required_columns:
        if col not in integral_table.columns:
            raise ValueError(f"Column '{col}' not found in the integral_table.")

    # Filter for Activators and Repressors
    activators = integral_table[
        (integral_table['JA_integral'] > 50) &
        (integral_table['SA_integral'] < -50) &
        (integral_table['SAJA_integral'] > 50) &
        (integral_table['gene'] != highlight_gene)  # Exclude the highlight_gene
    ]
    repressors = integral_table[
        (integral_table['JA_integral'] < 50) &
        (integral_table['SA_integral'] > 50) &
        (integral_table['SAJA_integral'] < -50)
    ]

    # Get the colormap and determine the max and min colors
    cmap = get_cmap('RdBu_r')
    activator_color = cmap(1.0)  # Max value of the colormap
    repressor_color = cmap(0.0)  # Min value of the colormap
    
    # Plot
    plt.figure(figsize=(10, 10))
    plt.xscale('symlog', linthresh=10)
    plt.yscale('symlog', linthresh=10)

    # Scatter plot for all genes
    scatter = plt.scatter(
        integral_table['JA_integral'],
        integral_table['SA_integral'],
        c=integral_table['SAJA_integral'],  # Use raw SAJA values
        cmap='RdBu_r',       # Use a colormap
        alpha=0.7,
        edgecolor='k',
        norm=SymLogNorm(linthresh=10,  vmin=-integral_table['SAJA_integral'].abs().max(),
                        vmax=integral_table['SAJA_integral'].abs().max())  # Apply symlog transformation
)

    # Highlight the specific gene if it exists
    if highlight_gene in integral_table['gene'].values:
        highlight_row = integral_table[integral_table['gene'] == highlight_gene]
        plt.scatter(
            highlight_row['JA_integral'],
            highlight_row['SA_integral'],
            c='none',  # No fill color
            edgecolor="black",  # black border
            linewidth=1.5,  # Thickness of the border
            s=100,  # Size of the marker
            label=f"Highlighted Gene: {highlight_gene}"
        )
    else:
        print(f"Warning: {highlight_gene} not found in the dataset. Skipping highlight.")

    # Check if legend exists
    legend_texts = []
    legend = plt.gca().get_legend()
    if legend:
        legend_texts = [t.get_text() for t in legend.get_texts()]

    # Circle each activator
    for _, row in activators.iterrows():
        plt.scatter(
            row['JA_integral'],
            row['SA_integral'],
            c='none',  # No fill color
            edgecolor=activator_color,  # Blue border for activators
            linewidth=1.5,  # Thickness of the border
            s=150,  # Size of the marker
            label="Activator" if 'Activator' not in legend_texts else None
        )

    # Circle each repressor
    for _, row in repressors.iterrows():
        plt.scatter(
            row['JA_integral'],
            row['SA_integral'],
            c='none',  # No fill color
            edgecolor=repressor_color,  # Green border for repressors
            linewidth=1.5,  # Thickness of the border
            s=150,  # Size of the marker
            label="Repressor" if 'Repressor' not in legend_texts else None
        )

    # Add reference lines
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    
    plt.grid(which='major', linestyle='--', alpha=0.7)

    # Customize ticks for symlog scale
    def custom_symlog_ticks(axis):
        """Generate custom ticks for a symlog scale."""
        # Define major ticks
        major_ticks = [-100000, -10000, -1000, -100, -10, 0, 10, 100, 1000, 10000, 100000]
        axis.set_major_locator(ticker.FixedLocator(major_ticks))

        # Define minor ticks for both positive and negative logarithmic regions
        minor_ticks = []
        for base in [10, 100, 1000, 10000, 100000]:
            minor_ticks.extend([-base * i for i in np.arange(1, 10)])
            minor_ticks.extend([base * i for i in np.arange(1, 10)])

        # Add minor ticks
        axis.set_minor_locator(ticker.FixedLocator(minor_ticks))
            # Format major ticks as powers of 10
        def format_func(value, _):
            if value == 0:
                return "0"
            exponent = int(np.log10(abs(value)))
            return f"$-10^{{{exponent}}}$" if value < 0 else f"$10^{{{exponent}}}$"

        axis.set_major_formatter(ticker.FuncFormatter(format_func))
    
    # Apply custom ticks to both axes
    custom_symlog_ticks(plt.gca().xaxis)
    custom_symlog_ticks(plt.gca().yaxis)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=plt.gca(), orientation='horizontal', pad=0.1)  # Explicitly associate the colorbar with the current Axes
    cbar.set_label("Cumulative Expression (Jasmonic & Salicylic acid)")

    # Customize plot
    plt.title(title)
    #plt.suptitle("Most transcription factors of MYC2 do not show jasmonic and salycylic acid antagonism.\n" \
    #"Gene selection: MYC2 DAPseq (add source) · Gene expression: RNAseq (add source)", fontsize=10, y=0.95)
    plt.xlabel("Cumulative Expression (Jasmonic acid)")
    plt.ylabel("Cumulative\nExpression\n(Salicylic acid)", rotation=0, labelpad=20)
    plt.tight_layout()
    #plt.show()

# Apply the function to the JA and SA datasets
plot_final_integral_expression_comparison_with_circles(
    integral_table,
    "Cumulative expression of MYC2 transcription factors encovers potential MYC2 activators and repressors",
    highlight_gene="AT1G32640"
)

# Save the plot as a PNG file
output_path = "c:/Users/sergej.pracek/OneDrive - NIB/Desktop/SKM/DAP-seq+RNAseq/Vizualizacija 3 - RNA + DAPseq MYC2/Expression_tf_of_MYC2_JA_SA_SAJA.png"
plt.savefig(output_path, format='png', dpi=600)  # Save with high resolution

print(f"Plot saved to {output_path}")
# Close the plot to free up memory
plt.close()


# Print all activators and repressors
# Filter for Activators and Repressors
activators = integral_table[
    (integral_table['JA_integral'] > 50) &
    (integral_table['SA_integral'] < -50) &
    (integral_table['SAJA_integral'] > 50) &
    (integral_table['gene'] != "AT1G32640")  # Exclude the highlight_gene
]
repressors = integral_table[
    (integral_table['JA_integral'] < 50) &
    (integral_table['SA_integral'] > 50) &
    (integral_table['SAJA_integral'] < -50)
]

print("Activators:")
print(activators[['gene', 'JA_integral', 'SA_integral', 'SAJA_integral']])

print("Repressors:")
print(repressors[['gene', 'JA_integral', 'SA_integral', 'SAJA_integral']])