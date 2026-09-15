from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data as DATA
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import os
from torch_geometric.data import DataLoader
import torch.nn.functional as F
# from torch.utils.data import DataLoader, Dataset

import re
class DTADataset(InMemoryDataset):
    def __init__(self, root, path, smiles_emb, target_emb, smiles_idx, smiles_graph, target_graph, smiles_len, target_len):

        super(DTADataset, self).__init__(root)
        self.path = path
        df = pd.read_csv(path)
        self.data = []
        self.process(df, smiles_emb, target_emb, smiles_idx, smiles_graph, target_graph, smiles_len, target_len)



    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['process.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, df, smiles_emb, target_emb, smiles_idx, smiles_graph, target_graph, smiles_len, target_len):
        # data_list = []
        for i in tqdm(range(len(df))):
            sm = df.loc[i, 'compound_iso_smiles']
            target = df.loc[i, 'target_key']
            seq = df.loc[i, 'target_sequence']
            label = df.loc[i, 'affinity']
            sm_g = smiles_graph[sm]
            ta_g = target_graph[target]
            sm_idx = smiles_idx[sm]
            tar_len = target_len[seq]

            s_off = self.off_adj(sm_g, tar_len)
            com_adj = np.concatenate((ta_g, s_off), axis=0)
            total_len = tar_len + len(sm_idx)
            tem1 = np.zeros([total_len, 2])
            tem2 = np.zeros([total_len, 2])
            for i in range(total_len):
                tem1[i, 0] = total_len
                tem1[i, 1] = i
                tem2[i, 1] = total_len
                tem2[i, 0] = i
            tem1 = np.int64(tem1)
            tem2 = np.int64(tem2)
            com_adj = np.concatenate((com_adj, tem1), axis=0)
            com_adj = np.concatenate((com_adj, tem2), axis=0)
            com_adj = np.concatenate((com_adj, [[total_len, total_len]]), axis=0)

            smiles = smiles_emb[sm]
            protein = target_emb[seq]
            smiles_lengths = smiles_len[sm]
            protein_lengths = target_len[seq]

            # smiles[i] = smiles_emb[sm]
            # protein[i] = target_emb[seq]
            # smiles_lengths.append(smiles_len[sm])
            # protein_lengths.append(target_len[seq])
            Data = DATA(y=torch.FloatTensor([label]),
                        edge_index=torch.LongTensor(com_adj).transpose(1, 0),
                        sm=sm,
                        target=target,
                        smiles=smiles,
                        protein=protein,
                        smiles_lengths=smiles_lengths,
                        protein_lengths=protein_lengths,
                        seq = seq
                        )

            ta_g = DATA(edge_index=torch.LongTensor(ta_g).transpose(1, 0),
                        )
            self.data.append((Data, ta_g))
        if self.pre_filter is not None:
            self.data = [data for data in self.data if self.pre_filter(self.data)]

        if self.pre_transform is not None:
            self.data = [self.pre_transform(data) for data in self.data]


    def off_adj(self, adj, size):
        adj1 = adj.copy()
        for i in range(adj1.shape[0]):
            adj1[i][0] += size
            adj1[i][1] += size
        return adj1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class DTA_Dataset(InMemoryDataset):
    def __init__(self, root, path, smiles_emb, target_emb, smiles_idx, smiles_graph, target_graph, smiles_len, target_len,mode, graph_dir=None, row_indices=None):

        super(DTA_Dataset, self).__init__(root)
        self.path = path
        df = pd.read_csv(path)
        if row_indices is not None:
            # Preserve caller-provided order (used for deterministic nested
            # subsets), then reset because process accesses rows by position.
            df = df.iloc[row_indices].reset_index(drop=True)
        self.mode = mode
        # Precomputed 3-D graphs are dataset-specific.  Keeping this explicit
        # avoids silently assuming the Davis ``pyg_8`` layout for KIBA.
        self.graph_dir = graph_dir or f'./{mode}/pyg_8'
        # A KIBA split reuses the same protein and compound structures across
        # many interactions. Cache immutable, precomputed PyG objects within
        # this dataset rather than deserializing a duplicate for every row.
        self._graph_cache = {}
        self.data = []
        sm_id = pd.read_csv("kiba/sm_id.csv")
        self.sm_id = {sm_id.loc[i, 'smiles']: sm_id.loc[i, 'id'] for i in range(len(sm_id))}
        self.process(df, smiles_emb, target_emb, smiles_idx, smiles_graph, target_graph, smiles_len, target_len)



    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['process.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, df, smiles_emb, target_emb, smiles_idx, smiles_graph, target_graph, smiles_len, target_len):
        # data_list = []
        if 'uniprot' in df.columns:
            data = "davis"
            df['id'] = df['uniprot']
        else:
            data = "kiba"
            df['id'] = df['target_key']
        # drug_emb = np.load(f'./{data}/{self.mode}/drug.npz',allow_pickle=True)
        # protein_emb = np.load(f'./{data}/{self.mode}/protein.npz', allow_pickle=True)
        drug_path = f'./{data}/default/drug.npz'
        protein_path = f'./{data}/default/protein.npz'
        # ``Data.x`` is legacy combined-graph input and is not consumed by
        # TriM-DTA/DMFF. Some released KIBA packages do not include these two
        # unused archives, so do not make the active three-branch model depend
        # on them merely to construct a dataset.
        has_legacy_embeddings = os.path.exists(drug_path) and os.path.exists(protein_path)
        if has_legacy_embeddings:
            drug_emb = {sm: torch.tensor(emb) for sm, emb in np.load(drug_path, allow_pickle=True).items()}
            protein_emb = {target: torch.tensor(emb) for target, emb in np.load(protein_path, allow_pickle=True).items()}
        for i in tqdm(range(len(df))):
            sm = df.loc[i, 'compound_iso_smiles']
            target = df.loc[i, 'target_key']
            seq = df.loc[i, 'target_sequence']
            label = df.loc[i, 'affinity']
            id = df.loc[i, 'id']
            sm_g = smiles_graph[sm]
            ta_g = target_graph[target]
            sm_idx = smiles_idx[sm]
            tar_len = target_len[seq]

            s_off = self.off_adj(sm_g, tar_len)
            com_adj = np.concatenate((ta_g, s_off), axis=0)
            total_len = tar_len + len(sm_idx)
            tem1 = np.zeros([total_len, 2])
            tem2 = np.zeros([total_len, 2])
            for i in range(total_len):
                tem1[i, 0] = total_len
                tem1[i, 1] = i
                tem2[i, 1] = total_len
                tem2[i, 0] = i
            tem1 = np.int64(tem1)
            tem2 = np.int64(tem2)
            com_adj = np.concatenate((com_adj, tem1), axis=0)
            com_adj = np.concatenate((com_adj, tem2), axis=0)
            com_adj = np.concatenate((com_adj, [[total_len, total_len]]), axis=0)

            smiles = smiles_emb[sm]
            protein = target_emb[seq]
            smiles_lengths = smiles_len[sm]
            protein_lengths = target_len[seq]


            Data = DATA(y=torch.FloatTensor([label]),
                        edge_index=torch.LongTensor(com_adj).transpose(1, 0),
                        smiles=smiles,
                        protein=protein,
                        smiles_lengths=smiles_lengths,
                        protein_lengths=protein_lengths,
                        )

            if has_legacy_embeddings:
                bs = (protein_emb[target][0].unsqueeze(0)+drug_emb[sm][0].unsqueeze(0))/2
                bs = F.pad(bs, (0,1), 'constant', 2)
                ts = F.pad(protein_emb[target], (0, 1), 'constant', 0)
                ss = F.pad(drug_emb[sm], (0,1), 'constant', 1)
                Data.x = torch.cat((ts, ss, bs), dim=0)
            target_graph_path = os.path.join(self.graph_dir, f'{id}.pt')
            sm_graph_name = sm if data == 'davis' else str(self.sm_id[sm])
            smiles_graph_path = os.path.join(self.graph_dir, f'{sm_graph_name}.pt')
            if target_graph_path not in self._graph_cache:
                self._graph_cache[target_graph_path] = torch.load(target_graph_path)
            if smiles_graph_path not in self._graph_cache:
                self._graph_cache[smiles_graph_path] = torch.load(smiles_graph_path)
            t_data = self._graph_cache[target_graph_path]
            s_data = self._graph_cache[smiles_graph_path]
            Data = (Data, s_data, t_data)
            self.data.append(Data)
        if self.pre_filter is not None:
            self.data = [data for data in self.data if self.pre_filter(self.data)]

        if self.pre_transform is not None:
            self.data = [self.pre_transform(data) for data in self.data]


    def off_adj(self, adj, size):
        adj1 = adj.copy()
        for i in range(adj1.shape[0]):
            adj1[i][0] += size
            adj1[i][1] += size
        return adj1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
