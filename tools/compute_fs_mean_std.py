import os
from typing import List, Union
import pandas as pd
from tabulate import tabulate


def aggregate_csv_metrics(
    data_root: Union[str, Path],
    csv_files: List[str],
    preserve_order: bool = True
) -> pd.DataFrame:
    """
    Batch reads CSV result files from a directory, computes mean and standard deviation
    for each metric per class (Name), and returns aggregated results with "mean ± std" format.

    Args:
        data_root: Root directory containing CSV files
        csv_files: List of CSV filenames to process
        preserve_order: Whether to preserve the original appearance order of classes

    Returns:
        Aggregated DataFrame with columns: Name, I-AUROC, I-AP, ..., each formatted as "mean ± std"
    """
    dfs = []

    for filename in csv_files:
        file_path = Path(data_root) / filename
        df = pd.read_csv(file_path)
        dfs.append(df)

    # Combine all DataFrames (preserves intra-file order)
    combined_df = pd.concat(dfs, ignore_index=True)

    # Record original order of unique Names (first appearance order)
    if preserve_order:
        original_order = combined_df["Name"].drop_duplicates(keep='first').tolist()

    # Identify metric columns (exclude 'Name')
    metric_cols = [col for col in combined_df.columns if col != "Name"]

    # Group by class name and compute mean and std separately
    grouped_mean = combined_df.groupby("Name")[metric_cols].mean()
    grouped_std = combined_df.groupby("Name")[metric_cols].std()

    # Round to one decimal place
    grouped_mean = grouped_mean.round(1)
    grouped_std = grouped_std.round(1)

    # Merge mean and std into "mean ± std" format
    merged_data = {}
    for col in metric_cols:
        merged_data[col] = (
            grouped_mean[col].astype(str) + " ± " + grouped_std[col].astype(str)
        )

    # Create new DataFrame
    result_df = pd.DataFrame(merged_data)
    result_df.reset_index(inplace=True)  # Adds 'Name' back as column

    # ✅ Preserve original row order (including Mean at the end)
    if preserve_order:
        result_df = result_df.set_index("Name").reindex(original_order).reset_index()

    return result_df

# ======================
# Main Execution
# ======================
if __name__ == "__main__":
    data_root = Path(f"./full-metric-results/12_4_128_train_on_mvtec_3learners_batch8/")
    data_root = Path(f"./results/12_4_128_train_on_mvtec_3learners_batch8/")

    seeds = [10, 20, 30]
    for test_dataset in ['RealIAD', 'Real-IAD-Variety', 'medical', 'medical-cls', 'visa']:  # ['mvtec', 'BTAD', 'MVTec3D', 'DTD-Synthetic', 'MPDD', 'SDD', 'medical-cls', 'medical-seg']
        for shot in [1, 2, 4]:
            output_csv = data_root / f"{test_dataset}_{shot}shot_mean_std.csv"

            # Collect CSV files
            csv_files_list = [
                f"{test_dataset}_{seed}seed_{shot}shot.csv"
                for seed in seeds
            ]

            # Process files and aggregate metrics
            aggregated_df = aggregate_csv_metrics(
                data_root=data_root,
                csv_files=csv_files_list,
                preserve_order=True
            )

            # Ensure output directory exists
            output_csv.parent.mkdir(parents=True, exist_ok=True)

            # Save results
            aggregated_df.to_csv(output_csv, index=False)
            print(f"✅ Aggregated results saved to: {output_csv.resolve()}")

            # Print as a nice table
            print(f"\nAggregated Results (Shot={shot}):")
            print(tabulate(aggregated_df, headers='keys', tablefmt='grid', showindex=False))
            print("\n" + "="*80 + "\n")

