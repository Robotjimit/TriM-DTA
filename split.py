import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def split(dataset: str, n_splits: int = 5) -> None:
    """
    Split the dataset into k-folds with different settings: warm start,
    drug cold start, protein cold start, and all cold start.
    """
    fpath = dataset + "/"
    df = pd.read_csv(fpath + f'{dataset}_processed.csv')

    # Apply log transformation to affinity if dataset is 'davis'
    # if dataset == "davis":
    #     merged_df["affinity"] = -np.log10(merged_df["affinity"] / 1e9)

    # Map drugs and proteins to unique integer IDs
    drug_ids = list(df["compound_iso_smiles"].unique())
    prot_ids = list(df["target_sequence"].unique())
    drug_map = {smiles: i for i, smiles in enumerate(drug_ids)}
    prot_map = {seq: i for i, seq in enumerate(prot_ids)}

    # Create drug-protein interaction dataframe with index
    dti = df[["compound_iso_smiles", "target_sequence", "affinity","target_key"]]
    if "uniprot" in df.columns:
        dti["uniprot"] = df["uniprot"]
    else:
        dti["uniprot"] = df["target_key"]

    dti["drug_idx"] = dti["compound_iso_smiles"].map(drug_map)
    dti["prot_idx"] = dti["target_sequence"].map(prot_map)
    # dti = dti[["compound_iso_smiles", "target_sequence","target_key", "affinity","uniprot"]].dropna().astype(float).reset_index(drop=True)

    def save_fold(data: pd.DataFrame, idx: int, name: str, setting: str) -> None:
        """Save the data fold to a CSV file."""

        fold_path = os.path.join(fpath, "data_folds", setting)
        os.makedirs(fold_path, exist_ok=True)
        data.to_csv(os.path.join(fold_path, f"{name}_fold_{idx}.csv"), index=None)

    def split_warm(setting: str = "warm_start") -> None:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        for idx, (train_index, test_index) in enumerate(kf.split(dti)):
            dti_train = dti.iloc[train_index]
            dti_test = dti.iloc[test_index]
            save_fold(dti_train, idx, "train", setting)
            save_fold(dti_test, idx, "test", setting)

    def split_drug_cold(setting: str = "drug_coldstart") -> None:
        drug_arr = np.array(range(len(drug_ids)))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        for idx, (train_index, test_index) in enumerate(kf.split(drug_arr)):
            train_drugs, test_drugs = drug_arr[train_index], drug_arr[test_index]
            dti_train = dti[dti["drug_idx"].isin(train_drugs)]
            dti_test = dti[dti["drug_idx"].isin(test_drugs)]
            save_fold(dti_train, idx, "train", setting)
            save_fold(dti_test, idx, "test", setting)

    def split_protein_cold(setting: str = "protein_coldstart") -> None:
        prot_arr = np.array(range(len(prot_ids)))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        for idx, (train_index, test_index) in enumerate(kf.split(prot_arr)):
            train_prots, test_prots = prot_arr[train_index], prot_arr[test_index]
            dti_train = dti[dti["prot_idx"].isin(train_prots)]
            dti_test = dti[dti["prot_idx"].isin(test_prots)]
            save_fold(dti_train, idx, "train", setting)
            save_fold(dti_test, idx, "test", setting)

    def split_all_cold(setting: str = "all_coldstart") -> None:
        drug_arr = np.array(range(len(drug_ids)))
        prot_arr = np.array(range(len(prot_ids)))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        drug_folds = list(kf.split(drug_arr))
        prot_folds = list(kf.split(prot_arr))

        for idx in range(n_splits):
            drug_train, drug_test = drug_arr[drug_folds[idx][0]], drug_arr[drug_folds[idx][1]]
            prot_train, prot_test = prot_arr[prot_folds[idx][0]], prot_arr[prot_folds[idx][1]]

            dti_train = dti[dti["drug_idx"].isin(drug_train) & dti["prot_idx"].isin(prot_train)]
            dti_test = dti[dti["drug_idx"].isin(drug_test) & dti["prot_idx"].isin(prot_test)]

            save_fold(dti_train, idx, "train", setting)
            save_fold(dti_test, idx, "test", setting)

    split_warm()
    split_drug_cold()
    split_protein_cold()
    split_all_cold()
    print(f"{dataset} dataset split completed.")


if __name__ == "__main__":
    for dataset in ["davis", "kiba"]:
        split(dataset)
