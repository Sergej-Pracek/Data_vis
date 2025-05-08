# load data RNAseq for 3 treatments
# Calculate cumulative expression during treatment
# Plot the cumulative expression for each TF in each treatment
# with the possibility to highlight a specific gene
# and circle activators and repressors

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.colors import SymLogNorm
from matplotlib import rcParams

# Import data with MultiIndex
df_JA = pd.read_csv(
    "RNA_seq_time_series_MCR_normalized_ordered_JA.csv",
    index_col=0,  # Use the first column as the index (gene names)
    header=[0, 1, 2]  # Use the first three rows as the MultiIndex for columns
)
df_ABA = pd.read_csv(
    "RNA_seq_time_series_MCR_normalized_ordered_ABA.csv",
    index_col=0,  # Use the first column as the index (gene names)
    header=[0, 1, 2]  # Use the first three rows as the MultiIndex for columns
)
df_ABAJA = pd.read_csv(
    "RNA_seq_time_series_MCR_normalized_ordered_ABAJA.csv",
    index_col=0,  # Use the first column as the index (gene names)
    header=[0, 1, 2]  # Use the first three rows as the MultiIndex for columns
)

# Load the data from the txt file
gene_list = "List_of_plotted_genes.txt"
df_gene = pd.read_csv(gene_list, sep="\t")

# Filter for only unique genes in the dataset
unique_genes = (df_gene['Gene']).unique()

# Filter the RNA-seq datasets
df_JA_subset = df_JA[df_JA.index.isin(unique_genes)]
df_ABA_subset = df_ABA[df_ABA.index.isin(unique_genes)]
df_ABAJA_subset = df_ABAJA[df_JA.index.isin(unique_genes)]

# Verify the subsets
print(f"Subset of df_JA: {df_JA_subset.shape}")
print(f"Subset of df_ABA: {df_ABA_subset.shape}")
print(f"Subset of df_ABAJA: {df_ABAJA_subset.shape}")

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
df_ABA_subset = reorder_by_timepoint(df_ABA_subset)
df_ABAJA_subset = reorder_by_timepoint(df_ABAJA_subset)

# Verify the reordering
print(f"Reordered df_JA_subset columns: {df_JA_subset.columns}")
print(f"Reordered df_ABA_subset columns: {df_ABA_subset.columns}")
print(f"Reordered df_ABAJA_subset columns: {df_ABAJA_subset.columns}")

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
df_ABA_with_integral = calculate_integral_between_curves(df_ABA_subset, "ABA")
df_ABAJA_with_integral = calculate_integral_between_curves(df_ABAJA_subset, "ABAJA")

# Display a sample of the updated DataFrame
print(df_JA_with_integral[[f"JA_vs_Mock_integral"]].head())
print(df_ABA_with_integral[[f"ABA_vs_Mock_integral"]].head())
print(df_ABAJA_with_integral[[f"ABAJA_vs_Mock_integral"]].head())

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
    (df_ABA_with_integral, "ABA"),
    (df_ABAJA_with_integral, "ABAJA")
]

top_genes = get_top_genes_with_min_absolute_value(datasets, top_n=5)

# Print the results
for label, genes in top_genes.items():
    print(f"{label} - Top 5 Genes with Lowest Absolute Min Values:")
    for gene, value in genes.items():
        print(f"  Gene: {gene}, Value: {value}")
# As some genes have cumulative expression = 0
# it is not possible to plot them on a log scale
# I will use symlog scale with linthresh = 10 for plotting

# creation of table with all genes and their integral values
def create_integral_table(df_JA, df_ABA, df_ABAJA):
    """
    Create a new dataset containing treatment_vs_mock_integral values for each treatment (JA, SA, SAJA) for each gene.

    Parameters:
    - df_JA: DataFrame containing the JA_vs_Mock_integral column.
    - df_ABA: DataFrame containing the ABA_vs_Mock_integral column.
    - df_ABAJA: DataFrame containing the ABAJA_vs_Mock_integral column.

    Returns:
    - A new DataFrame with columns: ['Gene', 'JA_integral', 'ABA_integral', 'ABAJA_integral'].
    """
    # Extract the integral columns
    ja_integral = df_JA["JA_vs_Mock_integral"].rename("JA_integral")
    aba_integral = df_ABA["ABA_vs_Mock_integral"].rename("ABA_integral")
    abaja_integral = df_ABAJA["ABAJA_vs_Mock_integral"].rename("ABAJA_integral")

    # Combine the integral columns into a single DataFrame
    combined_integral = pd.concat([ja_integral, aba_integral, abaja_integral], axis=1)

    # Reset the index to include the gene names as a column
    combined_integral.reset_index(inplace=True)
    combined_integral.rename(columns={"index": "gene"}, inplace=True)

    return combined_integral

# Create the new dataset
integral_table = create_integral_table(df_JA_with_integral, df_ABA_with_integral, df_ABAJA_with_integral)

# Display the first few rows of the new dataset
print(integral_table.head())

# Set the font to Lato
rcParams['font.family'] = 'Lato'

def plot_final_integral_expression_comparison_with_circles(integral_table, title, highlight_gene="AT1G32640"):
    """
    Plot the integral expression comparison between JA and ABA datasets.
    Colors are based on the logarithmic values of ABAJA_integral,
    and highlight a specific selected gene. Each activator and repressor is circled individually.

    Parameters:
    - integral_table: DataFrame containing columns ['gene', 'JA_integral', 'ABA_integral', 'ABAJA_integral'].
    - title: Title of the plot.
    - highlight_gene: Gene to highlight specifically (default: "AT1G32640" - MYC2).
    """
    # Ensure the required columns exist
    required_columns = ['gene', 'JA_integral', 'ABA_integral', 'ABAJA_integral']
    for col in required_columns:
        if col not in integral_table.columns:
            raise ValueError(f"Column '{col}' not found in the integral_table.")

    # Filter for Activators and Repressors
    activators = integral_table[
        (integral_table['JA_integral'] > 50) &
        (integral_table['ABA_integral'] < -50) &
        (integral_table['ABAJA_integral'] > 50) &
        (integral_table['gene'] != highlight_gene)  # Exclude the highlight_gene
    ]
    repressors = integral_table[
        (integral_table['JA_integral'] < 50) &
        (integral_table['ABA_integral'] > 50) &
        (integral_table['ABAJA_integral'] < -50)
    ]

    # Plot
    plt.figure(figsize=(8, 8))
    plt.xscale('symlog', linthresh=10)
    plt.yscale('symlog', linthresh=10)

    # Scatter plot for all genes
    scatter = plt.scatter(
        integral_table['JA_integral'],
        integral_table['ABA_integral'],
        c=integral_table['ABAJA_integral'],  # Use raw SAJA values
        cmap='RdBu_r',       # Use a colormap
        alpha=0.7,
        edgecolor='k',
        norm=SymLogNorm(linthresh=10,  vmin=-integral_table['ABAJA_integral'].abs().max(),
                        vmax=integral_table['ABAJA_integral'].abs().max())  # Apply symlog transformation
                        )

    # Highlight the specific gene if it exists
    if highlight_gene in integral_table['gene'].values:
        highlight_row = integral_table[integral_table['gene'] == highlight_gene]
        plt.scatter(
            highlight_row['JA_integral'],
            highlight_row['ABA_integral'],
            c='none',  # No fill color
            edgecolor="red",  # Red border
            linewidth=2,  # Thickness of the border
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
            row['ABA_integral'],
            c='none',  # No fill color
            edgecolor="blue",  # Blue border for activators
            linewidth=1.5,  # Thickness of the border
            s=150,  # Size of the marker
            label="Activator" if 'Activator' not in legend_texts else None
        )

    # Circle each repressor
    for _, row in repressors.iterrows():
        plt.scatter(
            row['JA_integral'],
            row['ABA_integral'],
            c='none',  # No fill color
            edgecolor="green",  # Green border for repressors
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
        axis.set_major_formatter(ticker.ScalarFormatter())

    # Apply custom ticks to both axes
    custom_symlog_ticks(plt.gca().xaxis)
    custom_symlog_ticks(plt.gca().yaxis)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=plt.gca(), orientation='horizontal', pad=0.1)  # Explicitly associate the colorbar with the current Axes
    cbar.set_label("Cumulative Expression (Jasmonic & Abscisic acid)")

    # Customize plot
    plt.title(title)
    plt.xlabel("Cumulative Expression (Jasmonic acid)")
    plt.ylabel("Cumulative\nExpression\n(Abscisic acid)", rotation=0, labelpad=20)
    plt.tight_layout()
    plt.show()

# Apply the function to the JA and SA datasets
plot_final_integral_expression_comparison_with_circles(
    integral_table,
    "Expression Comparison: Jasmonic vs Abscisic acid",
    highlight_gene="AT1G32640"
)