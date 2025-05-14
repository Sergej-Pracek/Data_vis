# load data RNAseq for 3 treatments
# Calculate cumulative expression during treatment
# Plot the cumulative expression for each TF in each treatment
# with the possibility to highlight a specific gene
# and circle activators and repressors
import sys
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.colors import SymLogNorm
from matplotlib import rcParams

# load local functions
from utils import load_RNAseq_data_file, calculate_integral_between_curves, create_integral_table

# Set the font to Lato
rcParams['font.family'] = 'Lato'

########################
# Functions
#######################



# To determine linear treshold for the symmetric log transformation, we can use the absolute minimum value
# from the integral values. This will help us to set the linthresh parameter for the symlog transformation.
# A rule of thumb: lintresh cca.=  min |X|
def get_top_genes_with_min_absolute_value(integral_table, treatments, top_n=5):
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
    for treatment in treatments:
        integral_col = f"{treatment}_integral"
        if integral_col in integral_table.columns:
            # Get the top N genes with the lowest absolute values
            abs_min_genes = integral_table[integral_col].abs().nsmallest(top_n).index
            abs_min_values = integral_table[integral_col].loc[abs_min_genes]
            top_genes[treatment] = abs_min_values.to_dict()
        else:
            print(
                f"Warning: {integral_col} not found in the table."
            )
    return top_genes

def prepare_RNAseq_data(datasets, gene_list):
    """
    Returns:
    - A DataFrame with columns: ['Gene', 'JA_integral', 'ABA_integral', 'ABAJA_integral'].
    """
    intergrals = []
    for treatment, file_path in datasets.items():
        df = load_RNAseq_data_file(file_path, gene_list)

        intergral = calculate_integral_between_curves(df, treatment)
        intergral.columns = [f"{treatment}_integral"]
        print(intergral.head(3))
        intergrals.append(intergral)

    # Combine the integral columns into a single DataFrame
    combined_integral = pd.concat(intergrals, axis=1)

    # Reset the index to include the gene names as a column
    combined_integral.reset_index(inplace=True)
    combined_integral.rename(columns={"index": "gene"}, inplace=True)

    print(combined_integral.head())

    return combined_integral

def prepare_gene_list(file_path):
    # Load the data from the txt file
    df_gene = pd.read_csv(file_path, sep="\t")

    # drop duplicated gene names
    df_gene.drop_duplicates(keep='first', inplace=True)

    #
    genes = df_gene['gene']
    highlight_genes = df_gene[df_gene['highlight'] == 1]['gene']
    gene_labels = df_gene.set_index("gene")["label"].to_dict()

    return genes, highlight_genes, gene_labels





def plot_final_integral_expression_comparison_with_circles(
    integral_table,
    treatments,
    title,
    highlight=False,
    highlight_genes=None,
    gene_labels=None,
    cutoff=50):
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
    required_columns = ['gene'] + [
        f'{treatment}_integral' for treatment in treatments
    ]

    for col in required_columns:
        if col not in integral_table.columns:
            raise ValueError(
                f"Column '{col}' not found in the integral_table.")

    x, xlabel = f'{treatments[0]}_integral', treatments[0]
    y, ylabel = f'{treatments[1]}_integral', treatments[1]
    z, zlabel = f'{treatments[2]}_integral', treatments[2]

    print(f"Plotting {x} (x) vs {y} (y) with color based on {z} (z). ")

    # Filter for Activators and Repressors
    activators = integral_table[
        (integral_table[x] > cutoff) & (integral_table[z] < -cutoff) &
        (integral_table[x] > cutoff)  #&
        # (integral_table['gene'] != highlight_gene)  # Exclude the highlight_gene
    ]
    repressors = integral_table[(integral_table[x] < cutoff)
                                & (integral_table[y] > cutoff) &
                                (integral_table[z] < -cutoff)]

    # Plot
    plt.figure(figsize=(8, 8))
    plt.xscale('symlog', linthresh=10)
    plt.yscale('symlog', linthresh=10)

    vlim = integral_table[z].abs().max()

    # Scatter plot for all genes
    scatter = plt.scatter(
        integral_table[x],
        integral_table[y],
        c=integral_table[z],  # Use raw SAJA values
        cmap='RdBu_r',  # Use a colormap
        alpha=0.7,
        edgecolor='k',
        norm=SymLogNorm(linthresh=10, vmin=-vlim,
                        vmax=vlim)  # Apply symlog transformation
    )

    # Highlight the specific gene if it exists
    if highlight:
        for highlight_gene in highlight_genes:
            if not (highlight_gene in integral_table['gene'].values):
                print(
                    f"Warning: {highlight_gene} not found in the dataset. Skipping highlight."
                )
            else:
                highlight_row = integral_table[integral_table['gene'] ==
                                               highlight_gene]
                plt.scatter(
                    highlight_row[x],
                    highlight_row[y],
                    c='none',  # No fill color
                    edgecolor="green",  # Green border
                    linewidth=2,  # Thickness of the border
                    s=100,  # Size of the marker
                    # label=f"Highlighted Gene: {highlight_gene}"
                    label=gene_labels[highlight_gene])
                plt.text(highlight_row[x],
                         highlight_row[y],
                         s="  " + gene_labels[highlight_gene])

    # Check if legend exists
    legend_texts = []
    legend = plt.gca().get_legend()
    if legend:
        legend_texts = [t.get_text() for t in legend.get_texts()]

    # Circle each activator
    for _, row in activators.iterrows():
        plt.scatter(
            row[x],
            row[y],
            c='none',  # No fill color
            edgecolor="red",  # Red border for activators
            linewidth=1.5,  # Thickness of the border
            s=150,  # Size of the marker
            label="Activator" if 'Activator' not in legend_texts else None)

    # Circle each repressor
    for _, row in repressors.iterrows():
        plt.scatter(
            row[x],
            row[y],
            c='none',  # No fill color
            edgecolor="blue",  # Blue border for repressors
            linewidth=1.5,  # Thickness of the border
            s=150,  # Size of the marker
            label="Repressor" if 'Repressor' not in legend_texts else None)

    # Add reference lines
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)

    plt.grid(which='major', linestyle='--', alpha=0.7)

    # Customize ticks for symlog scale
    def custom_symlog_ticks(axis):
        """Generate custom ticks for a symlog scale."""
        # Define major ticks
        major_ticks = [
            -100000, -10000, -1000, -100, -10, 0, 10, 100, 1000, 10000, 100000
        ]
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
    cbar = plt.colorbar(
        scatter, ax=plt.gca(), orientation='horizontal',
        pad=0.1)  # Explicitly associate the colorbar with the current Axes
    cbar.set_label(f"Cumulative Expression ({zlabel})") #TODO pretty label

    # Customize plot
    plt.title(title)
    plt.xlabel(f"Cumulative Expression ({xlabel})") #TODO pretty label
    plt.ylabel(f"Cumulative\nExpression\n({ylabel})", #TODO pretty label
               rotation=0,
               labelpad=20)
    plt.tight_layout()


########################
# MAIN SCRIPT
#######################


def main(datasets, gene_list_file, fig, highlight=True):
    """
    """

    treatments = list(datasets.keys())
    print(f"Treatments: {treatments}")

    # Load the gene list
    gene_list, highlight_genes, gene_labels = prepare_gene_list(gene_list_file)

    # Prepare the RNA-seq data for each treatment
    integral_table = prepare_RNAseq_data(datasets, gene_list)

    top_genes = get_top_genes_with_min_absolute_value(integral_table, treatments, top_n=5)

    # Print the results
    for label, genes in top_genes.items():
        print(f"{label} - Top 5 Genes with Lowest Absolute Min Values:")
        for gene, value in genes.items():
            print(f"  Gene: {gene}, Value: {value}")
    # As some genes have cumulative expression = 0
    # it is not possible to plot them on a log scale
    # I will use symlog scale with linthresh = 10 for plotting

    plot_final_integral_expression_comparison_with_circles(
        integral_table,
        treatments,
        "Expression Comparison: Jasmonic vs Abscisic acid",
        highlight=highlight,
        highlight_genes=highlight_genes,
        gene_labels=gene_labels)

    if fig:
        print(f"Saving to {fig}.")
        plt.savefig(fig, transparent=True, bbox_inches='tight')
    else:
        plt.show()


########################
# COMMANDLINE STUFF
#######################


def parse_key_value_args(arg):
    """
    Parse key-value argument passed in the format "key=value".
    """
    args = {}
    for entry in arg:
        try:
            key, value = entry.split('=')
            args[key] = value
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Argument '{arg}' is not in the format 'key1=value1 key2=value2 ...'.")

    return args


def arguments():

    parser = argparse.ArgumentParser(
        description=
        'Plot expression comparison between JA and ABA datasets, for subset of genes.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # required arguments
    parser.add_argument(
        "--datasets",
        action="extend", nargs="+",
        type=str,
        help=
        "Treatments and file locations in the form "
        "'treatment1=file_path1 treatment2=file_path2', e.g. "
        "ABAJA=./example/RNA_seq_time_series_MCR_normalized_ordered_ABAJA.csv "
        "ABA=./example/RNA_seq_time_series_MCR_normalized_ordered_ABA.csv ",
        required=True
    )

    parser.add_argument(
        "--gene_list",
        type=str,
        help=
        "File with list of genes, columns are 'gene', 'label', 'highlight'."
        "highlight = 1 means the gene will be highlighted in the plot (if --highlight is True).",
        required=True
    )

    parser.add_argument(
        "--figure",
        type=str,
        help="File name to save figure to. If not given, shows interactively.",
        default=None
    )

    # optional arguments
    parser.add_argument(
        "--highlight",
        type=bool,
        default=True,
        help="Highlight genes."
    )

    return parser


if __name__ == '__main__':
    parser = arguments()
    args = parser.parse_args()

    # get treatments and file paths
    datasets = parse_key_value_args(args.datasets)
    print("Using files:")
    for t, path in datasets.items():
        print(f"  {t}: {path}")

    # Apply the function to the JA and SA datasets
    main(datasets, args.gene_list, args.figure, args.highlight)