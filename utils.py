import pandas as pd
import numpy as np

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

def load_RNAseq_data_file(file_path, gene_list):
    """
    Load RNA-seq data from a CSV file.

    Parameters:
    - file_path: Path to the CSV file.

    Returns:
    - DataFrame containing the RNA-seq data.
    """

    # Import data with MultiIndex
    df = pd.read_csv(
        file_path,
        index_col=0,  # Use the first column as the index (gene names)
        header=[0, 1,
                2]  # Use the first three rows as the MultiIndex for columns
    )

    df = df[df.index.isin(gene_list)]
    print(f"Subset of df: {df.shape}")

    df = reorder_by_timepoint(df)
    print(f"Reordered df columns: {df.columns}")

    return df

# creation of table with all genes and their integral values
def create_integral_table(datasets):
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
    intergrals = []
    for treatment, df in datasets.items():
        integral_col = f"{treatment}_vs_Mock_integral"
        if integral_col in df.columns:
            intergrals.append(df[[integral_col
                                  ]].rename(f"{treatment}_integral"))

    # Combine the integral columns into a single DataFrame
    combined_integral = pd.concat(intergrals, axis=1)

    # Reset the index to include the gene names as a column
    combined_integral.reset_index(inplace=True)
    combined_integral.rename(columns={"index": "gene"}, inplace=True)

    return combined_integral


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
    treatment_cols = df.loc[:,
                            df.columns.get_level_values('treatment') ==
                            treatment_label]
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
    # df[f"{treatment_label}_vs_Mock_integral"] = integral_values
    # return df

    # Create a new DataFrame with gene names and integral values
    integral_df = pd.DataFrame(
        integral_values, index=corrected_df.index, columns=[f"{treatment_label}_vs_Mock_integral"]
    )

    return integral_df
