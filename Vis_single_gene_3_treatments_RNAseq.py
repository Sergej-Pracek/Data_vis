# load data RNAseq
# filter for JA, SA, SAJA treatments
# first plot MYC2 for JA, SA, SAJA treatments
# for each tf, determine if it is activated, not regulated, or repressed
# plot 3x3 drid of timeseries for each tf
# First line JA, second line SA, third line SAJA
# first column activated, second column not regulated, third column repressed

import pandas as pd
import matplotlib.pyplot as plt

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

# Visualize MYC2 (AT1G32640) for all three datasets

# Define the gene and its label
gene = 'AT1G32640'
gene_label = 'MYC2'
# Datasets and their labels
datasets = [(df_JA, "JA"), (df_ABA, "ABA"), (df_ABAJA, "ABAJA")]

# Option 1: Create a time series plot for each dataset
# Create a 1x3 grid for subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Loop through datasets and plot
for i, (df, treatment_label) in enumerate(datasets):
    ax = axes[i]

    # Filter data for the specific gene and treatments
    mock_data_gene = df["Mock"].loc[gene]
    treatment_data_gene = df[treatment_label].loc[gene]

    # Aggregate replicates for Mock
    mock_mean_values = mock_data_gene.groupby(level="timepoint").mean()
    mock_std_values = mock_data_gene.groupby(level="timepoint").std()

    # Aggregate replicates for the treatment
    treatment_mean_values = treatment_data_gene.groupby(level="timepoint").mean()
    treatment_std_values = treatment_data_gene.groupby(level="timepoint").std()

    # Convert the timepoints to numeric values for consistent x-axis handling
    timepoints = mock_mean_values.index.astype(float)
    timepoints_sorted_indices = timepoints.argsort()

    # Sort all timepoints and corresponding values for consistent ordering
    timepoints = timepoints[timepoints_sorted_indices]
    mock_mean_values = mock_mean_values.iloc[timepoints_sorted_indices]
    mock_std_values = mock_std_values.iloc[timepoints_sorted_indices]
    treatment_mean_values = treatment_mean_values.iloc[timepoints_sorted_indices]
    treatment_std_values = treatment_std_values.iloc[timepoints_sorted_indices]

    # Plot Mock mean line
    ax.plot(timepoints, mock_mean_values, label="Mock", color="blue")
    # Plot Mock shaded standard deviation
    ax.fill_between(
        timepoints,
        mock_mean_values - mock_std_values,
        mock_mean_values + mock_std_values,
        color="blue",
        alpha=0.2
    )

    # Plot treatment mean line
    ax.plot(timepoints, treatment_mean_values, label=treatment_label, color="green")
    # Plot treatment shaded standard deviation
    ax.fill_between(
        timepoints,
        treatment_mean_values - treatment_std_values,
        treatment_mean_values + treatment_std_values,
        color="green",
        alpha=0.2
    )

    # Set title and labels
    ax.set_title(f"{gene_label} ({gene}) - {treatment_label}")
    ax.set_xlabel("Timepoint [hpt]")
    if i == 0:
        ax.set_ylabel("Expression Level")
    ax.legend(fontsize=8)

# Final adjustments
plt.tight_layout()
plt.show()