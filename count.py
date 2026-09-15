
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm
def data_split():
    def split_csv_dataset(input_csv_path='davis_processed.csv', train_ratio=0.7, valid_ratio=0.1, test_ratio=0.2, seed=42):
        # 读取 CSV 文件
        df = pd.read_csv(input_csv_path)

        # 首先划分出 train 和 temp（valid + test）
        train_df, temp_df = train_test_split(df, test_size=(1 - train_ratio), random_state=seed)

        # 然后从 temp 中再划分出 valid 和 test
        valid_size = valid_ratio / (valid_ratio + test_ratio)
        valid_df, test_df = train_test_split(temp_df, test_size=(1 - valid_size), random_state=seed)

        # 保存为新的 CSV 文件
        train_indices = train_df.index.tolist()
        valid_indices = valid_df.index.tolist()
        test_indices = test_df.index.tolist()
        return {
            "train": train_indices,
            "valid": valid_indices,
            "test": test_indices,
        }

    def create_fold_setting_cold(df, fold_seed, frac, entities):
        """create cold-split where given one or multiple columns, it first splits based on
        entities in the columns and then maps all associated data points to the partition

        Args:
                df (pd.DataFrame): dataset dataframe
                fold_seed (int): the random seed
                frac (list): a list of train/valid/test fractions
                entities (Union[str, List[str]]): either a single "cold" entity or a list of
                        "cold" entities on which the split is done

        Returns:
                dict: a dictionary of splitted dataframes, where keys are train/valid/test and values correspond to each dataframe
        """
        if entities == 'target_key':
            entities = 'new'
            df['new'] = df['target_key'].astype(str)+'_'+df['target_sequence_start'].astype(str)+'_'+df['target_sequence_end'].astype(str)
        if isinstance(entities, str):
            entities = [entities]

        train_frac, val_frac, test_frac = frac

        # For each entity, sample the instances belonging to the test datasets
        test_entity_instances = [
            df[e].drop_duplicates().sample(frac=test_frac,
                                        replace=False,
                                        random_state=fold_seed).values
            for e in entities
        ]

        # Select samples where all entities are in the test set
        test = df.copy()
        for entity, instances in zip(entities, test_entity_instances):
            test = test[test[entity].isin(instances)]

        if len(test) == 0:
            raise ValueError(
                "No test samples found. Try another seed, increasing the test frac or a "
                "less stringent splitting strategy.")

        # Proceed with validation data
        train_val = df.copy()
        for i, e in enumerate(entities):
            train_val = train_val[~train_val[e].isin(test_entity_instances[i])]

        val_entity_instances = [
            train_val[e].drop_duplicates().sample(frac=val_frac / (1 - test_frac),
                                                replace=False,
                                                random_state=fold_seed).values
            for e in entities
        ]
        val = train_val.copy()
        for entity, instances in zip(entities, val_entity_instances):
            val = val[val[entity].isin(instances)]

        if len(val) == 0:
            raise ValueError(
                "No validation samples found. Try another seed, increasing the test frac "
                "or a less stringent splitting strategy.")

        train = train_val.copy()
        for i, e in enumerate(entities):
            train = train[~train[e].isin(val_entity_instances[i])]
        train_indices = df[df.isin(train)].dropna().index
        val_indices = df[df.isin(val)].dropna().index
        test_indices = df[df.isin(test)].dropna().index
        # return {
        #     "train": train.reset_index(drop=True),
        #     "valid": val.reset_index(drop=True),
        #     "test": test.reset_index(drop=True),
        # }
        return {
            "train": train_indices,
            "valid": val_indices,
            "test": test_indices,
        }
    dataname = 'kiba'
    df = pd.read_csv(f'{dataname}/{dataname}_processed.csv')
    split = create_fold_setting_cold(df, fold_seed=42, frac=[0.7, 0.1, 0.2], entities=['compound_iso_smiles'])
    # split = split_csv_dataset(input_csv_path=f'{dataname}/{dataname}_processed.csv', train_ratio=0.7, valid_ratio=0.1, test_ratio=0.2, seed=42)
    train_df = df.iloc[split['train']].reset_index(drop=True)
    valid_df = df.iloc[split['valid']].reset_index(drop=True)
    test_df  = df.iloc[split['test']].reset_index(drop=True)
    mode = ['default', 'drug_cold', 'target_cold', 'all_cold']
    mode_name = mode[1]
    # 保存为 CSV 文件
    train_df.to_csv(f'{dataname}/{mode_name}/fold0/train.csv', index=False)
    valid_df.to_csv(f'{dataname}/{mode_name}/fold0/valid.csv', index=False)
    test_df.to_csv(f'{dataname}/{mode_name}/fold0/test.csv', index=False)

def id_download():
    # 获取 target_key 列的唯一值
    unique_targets = df['target_key'].unique()
    targets_str = ','.join(unique_targets)
    # 保存为 txt 文件
    with open("uniprot.txt", "w") as f:
        f.write(targets_str)
    # 从 uniprot.txt 读取 ID 列表（用逗号分隔）
    with open("uniprot.txt", "r") as f:
        content = f.read()
        uniprot_ids = [uid.strip() for uid in content.split(",") if uid.strip()]
    import os
    import requests
    # 遍历 ID 并下载对应的 AlphaFold 结构
    pdb_folder = './kiba/pdb'
    for uid in tqdm(uniprot_ids):
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v4.pdb"
        response = requests.get(url)
        if response.status_code == 200:
            with open(os.path.join(pdb_folder, f"{uid}.pdb"), "w") as f:
                f.write(response.text)
            print(f"Downloaded {uid}.pdb")
        else:
            print(f"Failed to download {uid}: HTTP {response.status_code}")

data_split()
def ttest():
    from scipy.stats import ttest_rel

    trim_dta = [0.190, 0.195, 0.188, 0.192, 0.193]
    baseline = [0.225, 0.229, 0.220, 0.227, 0.222]

    t_stat, p_value = ttest_rel(baseline, trim_dta)
    print(f'p-value = {p_value:.4f}')
