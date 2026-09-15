import numpy as np
import pandas as pd
import rdkit
import rdkit.Chem as Chem
import networkx as nx

import torch
import os
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from build_vocab import WordVocab
from utils import *
from moe import *
from dataset import DTADataset
from sklearn.model_selection import KFold
from model import *
from torch import nn as nn

CUDA = '0'
device = torch.device('cuda:' + CUDA)
LR = 1e-3
NUM_EPOCHS = 100
seed = 0
dataset_name = 'davis'
batch_size = 128


#############################################################################


class DMFF(nn.Module):
    def __init__(self, embedding_dim: int, lstm_dim: int, hidden_dim: int, dropout_rate: float,
                 alpha: float, n_heads: int, bilstm_layers: int = 2, protein_vocab: int = 26,
                 smile_vocab: int = 45, theta: float = 0.5):
        """
        初始化 DMFF 模型。

        :param embedding_dim: 嵌入维度
        :param lstm_dim: LSTM 维度
        :param hidden_dim: 隐藏层维度
        :param dropout_rate: dropout 比率
        :param alpha: LeakyReLU 的 alpha 值
        :param n_heads: 注意力头的数量
        :param bilstm_layers: 双向 LSTM 层数
        :param protein_vocab: 蛋白质词汇表大小
        :param smile_vocab: SMILES 词汇表大小
        :param theta: 超参数
        """
        super(DMFF, self).__init__()
        self.theta = theta
        self.dropout = nn.Dropout(dropout_rate)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.bilstm_layers = bilstm_layers
        self.n_heads = n_heads

        # SMILES 相关
        self.smiles_vocab = smile_vocab
        self.smiles_embed = nn.Embedding(smile_vocab + 1, embedding_dim, padding_idx=0)
        # self.smiles_lstm = nn.LSTM(lstm_dim, lstm_dim, self.bilstm_layers, batch_first=True,
        #                            bidirectional=True, dropout=dropout_rate)
        # self.smiles_lstm = xLSTM(input_size=lstm_dim, head_size=lstm_dim, num_heads=2, layers=['m', 's', 'm'], batch_first=True)
        self.smiles_lstm = nn.GRU(lstm_dim, lstm_dim, self.bilstm_layers, batch_first=True,
                                    bidirectional=True, dropout=dropout_rate)
        self.smiles_conv = TCNBlock(embedding_dim)
        self.enhance1 = SpatialGroupEnhance_for_1D(groups=20)
        self.ln1 = torch.nn.LayerNorm(hidden_dim)

        # 蛋白质相关
        self.protein_vocab = protein_vocab
        self.protein_embed = nn.Embedding(protein_vocab + 1, embedding_dim, padding_idx=0)
        # self.protein_lstm = nn.LSTM(lstm_dim, lstm_dim, self.bilstm_layers, batch_first=True,
        #                             bidirectional=True, dropout=dropout_rate)
        # self.protein_lstm = xLSTM(input_size=lstm_dim, head_size=lstm_dim, num_heads=2, layers=['m', 's', 'm'], batch_first=True)
        self.protein_lstm = nn.GRU(lstm_dim, lstm_dim, self.bilstm_layers, batch_first=True,
                                    bidirectional=True, dropout=dropout_rate)
        self.protein_conv = TCNBlock(embedding_dim)
        self.enhance2 = SpatialGroupEnhance_for_1D(groups=200)
        self.ln2 = torch.nn.LayerNorm(hidden_dim)

        # 输出层
        self.out_fc1 = nn.Linear(hidden_dim * 3, 256 * 4)
        self.out_fc2 = nn.Linear(256 * 4, hidden_dim)
        self.out_fc3 = nn.Linear(hidden_dim, 1)


        # Point-wise Feed Forward Network
        self.pwff_1 = nn.Linear(hidden_dim * 3, hidden_dim * 4)
        self.pwff_2 = nn.Linear(hidden_dim * 4, hidden_dim * 3)

    def forward(self, data, reset=False):
        data, _, = data
        batchsize = len(data.sm)
        smiles = torch.zeros(batchsize, seq_len).to(device).long()
        protein = torch.zeros(batchsize, tar_len).to(device).long()
        smiles_lengths = []
        protein_lengths = []

        for i in range(batchsize):
            sm = data.sm[i]
            seq_id = data.target[i]
            seq = target_seq[seq_id]
            smiles[i] = smiles_emb[sm]
            protein[i] = target_emb[seq]
            smiles_lengths.append(smiles_len[sm])
            protein_lengths.append(target_len[seq])

        smiles = self.smiles_embed(smiles)  # B * seq len * emb_dim
        smiles = self.smiles_conv(smiles)
        smiles = self.enhance1(smiles)
        smiles, _ = self.smiles_lstm(smiles)  # B * seq len * lstm_dim*2
        smiles = self.ln1(smiles)

        protein = self.protein_embed(protein)  # B * tar_len * emb_dim
        protein = self.protein_conv(protein)
        protein = self.enhance2(protein)
        protein, _ = self.protein_lstm(protein)  # B * tar_len * lstm_dim *2
        protein = self.ln2(protein)

        if reset:
            return smiles, protein

        smiles_mask = self.generate_masks(smiles, smiles_lengths, self.n_heads)  # B * head* seq len
        protein_mask = self.generate_masks(protein, protein_lengths, self.n_heads)  # B * head * tar_len
        out_cat = torch.cat((smiles, protein), dim=1)  # B * head * lstm_dim *2
        out_masks = torch.cat((smiles_mask, protein_mask), dim=1)  # B * tar_len+seq_len * (lstm_dim *2)

        smiles_out = self.masked_mean_pooling(smiles, smiles_mask)  # B * lstm_dim*2
        protein_out = self.masked_mean_pooling(protein, protein_mask)  # B * (lstm_dim *2)
        out_cat = self.masked_mean_pooling(out_cat, out_masks)
        out = torch.cat([smiles_out, protein_out, out_cat], dim=-1)  # B * (rnn*2 *3)

        # Point-wise Feed Forward Network
        pwff = self.dropout(self.relu(self.pwff_1(out)))
        pwff = self.dropout(self.relu(self.pwff_2(pwff)))
        out = pwff + out

        out = self.dropout(self.relu(self.out_fc1(out)))  # B * (256*8)
        out = self.dropout(self.relu(self.out_fc2(out)))  # B *  hidden_dim*2

        out = self.out_fc3(out).squeeze()

        del smiles_out, protein_out

        return out, data.y

    def generate_masks(self, adj, adj_sizes, n_heads):
        out = torch.ones(adj.shape[0], adj.shape[1])
        max_size = adj.shape[1]
        if isinstance(adj_sizes, int):
            out[0, adj_sizes:max_size] = 0
        else:
            for e_id, drug_len in enumerate(adj_sizes):
                out[e_id, drug_len: max_size] = 0
        return out.cuda(device=adj.device)

    def masked_mean_pooling(self, x, mask):
        # x: [B, L, D], mask: [B, L]
        mask = mask.unsqueeze(-1)  # [B, L, 1]
        x = x * mask  # zero out padded positions
        sum_x = x.sum(dim=1)  # [B, D]
        lengths = mask.sum(dim=1).clamp(min=1e-6)  # [B, 1]
        return sum_x / lengths  # [B, D]


def smiles_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
    edge_index = np.array(edge_index)
    return c_size, edge_index


#############################################################################

df = pd.read_csv(f'./{dataset_name}/{dataset_name}_processed.csv')

smiles = set(df['compound_iso_smiles'])
target = set(df['target_key'])

target_seq = {}
for i in range(len(df)):
    target_seq[df.loc[i, 'target_key']] = df.loc[i, 'target_sequence']

smiles_graph = {}
for sm in smiles:
    _, graph = smiles_to_graph(sm)
    smiles_graph[sm] = graph

target_uniprot_dict = {}
target_process_start = {}
target_process_end = {}

for i in range(len(df)):
    target = df.loc[i, 'target_key']
    if dataset_name == 'kiba':
        uniprot = df.loc[i, 'target_key']
    else:
        uniprot = df.loc[i, 'uniprot']
    target_uniprot_dict[target] = uniprot
    target_process_start[target] = df.loc[i, 'target_sequence_start']
    target_process_end[target] = df.loc[i, 'target_sequence_end']

contact_dir = './target_contact_map_' + dataset_name + '/'
target_graph = {}


def target_to_graph(target_key, target_sequence, contact_dir, start, end):
    target_edge_index = []
    target_size = len(target_sequence)
    contact_file = os.path.join(contact_dir, target_key + '.npy')
    contact_map = np.load(contact_file)
    contact_map = contact_map[start:end, start:end]
    index_row, index_col = np.where(contact_map > 0.8)

    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])
    target_edge_index = np.array(target_edge_index)
    return target_size, target_edge_index


for target in tqdm(target_seq.keys()):
    uniprot = target_uniprot_dict[target]
    contact_map = np.load(contact_dir + uniprot + '.npy')
    start = target_process_start[target]
    end = target_process_end[target]
    _, graph = target_to_graph(uniprot, target_seq[target], contact_dir, start, end)
    target_graph[target] = graph

drug_vocab = WordVocab.load_vocab('./Vocab/smiles_vocab.pkl')
target_vocab = WordVocab.load_vocab('./Vocab/protein_vocab.pkl')

tar_len = 1000
seq_len = 540

smiles_idx = {}
smiles_emb = {}
smiles_len = {}
for sm in smiles:
    content = []
    flag = 0
    for i in range(len(sm)):
        if flag >= len(sm):
            break
        if (flag + 1 < len(sm)):
            if drug_vocab.stoi.__contains__(sm[flag:flag + 2]):
                content.append(drug_vocab.stoi.get(sm[flag:flag + 2]))
                flag = flag + 2
                continue
        content.append(drug_vocab.stoi.get(sm[flag], drug_vocab.unk_index))
        flag = flag + 1

    if len(content) > seq_len:
        content = content[:seq_len]

    X = [drug_vocab.sos_index] + content + [drug_vocab.eos_index]
    smiles_len[sm] = len(content)
    if seq_len > len(X):
        padding = [drug_vocab.pad_index] * (seq_len - len(X))
        X.extend(padding)

    smiles_emb[sm] = torch.tensor(X)

    if not smiles_idx.__contains__(sm):
        tem = []
        for i, c in enumerate(X):
            if atom_dict.__contains__(c):
                tem.append(i)
        smiles_idx[sm] = tem

target_emb = {}
target_len = {}
for k in target_seq:
    seq = target_seq[k]
    content = []
    flag = 0
    for i in range(len(seq)):
        if flag >= len(seq):
            break
        if (flag + 1 < len(seq)):
            if target_vocab.stoi.__contains__(seq[flag:flag + 2]):
                content.append(target_vocab.stoi.get(seq[flag:flag + 2]))
                flag = flag + 2
                continue
        content.append(target_vocab.stoi.get(seq[flag], target_vocab.unk_index))
        flag = flag + 1

    if len(content) > tar_len:
        content = content[:tar_len]

    X = [target_vocab.sos_index] + content + [target_vocab.eos_index]
    target_len[seq] = len(content)
    if tar_len > len(X):
        padding = [target_vocab.pad_index] * (tar_len - len(X))
        X.extend(padding)
    target_emb[seq] = torch.tensor(X)


def reset_feature_and_save(dataset, model, save_dir='davis/default'):
    os.makedirs(save_dir, exist_ok=True)
    torch.cuda.empty_cache()
    batch_size = 512

    bigraph_dict = {}
    drug_dict = {}
    protein_dict = {}
    with torch.no_grad():
        model.eval()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        start = 0
        for data in tqdm(dataloader):
            sm, pro = model(data, reset=True)
            tar_len = []
            idx = []

            for i in range(min(batch_size, len(dataset) - start)):
                data_i, sm_g, pro_g = dataset.data[start + i]
                sm_id = data_i.sm
                pro_id = data_i.target
                pro_id = target_seq[pro_id]
                tar_len.append(target_len[pro_id])
                idx.append(smiles_idx[sm_id])

            for i in range(start, min(len(dataset), start + batch_size)):
                data_i, sm_g, pro_g = dataset.data[i]
                t_len = tar_len[i - start]
                s_idx = idx[i - start]

                sm_g.x = sm[i - start, s_idx]

                pro_g.x = pro[i - start, 1:t_len + 1]
                pro_g.cpu()
                sm_g.cpu()
                data_i.cpu()
                prefix = os.path.join(save_dir, str(i + 1))  # i+1 保证从1开始编号
                if data_i.target not in protein_dict:
                    protein_dict[data_i.target] = pro_g.x.cpu().numpy()
                if data_i.sm not in drug_dict:
                    drug_dict[data_i.sm] = sm_g.x.cpu().numpy()

            start = start + batch_size
        # np.savez_compressed(os.path.join(save_dir, "bigraph.npz"), **bigraph_dict)
        np.savez_compressed(os.path.join(save_dir, "drug.npz"), **drug_dict)
        np.savez_compressed(os.path.join(save_dir, "protein.npz"), **protein_dict)


print("Building dataset...")
data_set = f"{dataset_name}_processed"
mode = ['warm_start','drug_coldstart','protein_coldstart','all_coldstart']
mode_name = mode[2]
seed = [18, 283, 839, 12, 74]

# dataset = DTADataset(root='./', path=f'./{dataset_name}/' + data_set + '.csv', smiles_emb=smiles_emb, target_emb=target_emb,
#                      smiles_idx=smiles_idx, smiles_graph=smiles_graph, target_graph=target_graph, smiles_len=smiles_len,
#                      target_len=target_len)



num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=0)
dim = 128
for fold in range(5):
    model_file_name = './Model/' + dataset_name + '_' + mode_name + '_fold_' + str(fold) + '.pt'
    print("Building model...")

    model = DMFF(embedding_dim=dim * 2, lstm_dim=dim, hidden_dim=dim * 2, dropout_rate=0.2,
                        alpha=0.2, n_heads=8, bilstm_layers=2, protein_vocab=26,
                        smile_vocab=45, theta=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # if model_file_name is not None:
    #     model.load_state_dict(torch.load(model_file_name), strict=False)
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=4e-5,verbose=True)
    best_mse = 1000
    best_test_mse = 1000
    best_epoch = -1
    best_test_epoch = -1

    print(f"Fold {fold + 1}")
    log(f'train on {dataset_name}_{mode_name}')
    for epoch in range(NUM_EPOCHS):
        print("No {} epoch".format(epoch))
        if epoch == 0:
            # reset_feature_and_save(dataset, model, save_dir=f'./{dataset_name}/{mode_name}')
            train_dataset = DTADataset(root='./', path=f'./{dataset_name}/data_folds/{mode_name}/train_fold_{fold}.csv', smiles_emb=smiles_emb,
                                       target_emb=target_emb, smiles_idx=smiles_idx, smiles_graph=smiles_graph,
                                       target_graph=target_graph, smiles_len=smiles_len, target_len=target_len)
            test_dataset = DTADataset(root='./', path=f'./{dataset_name}/data_folds/{mode_name}/test_fold_{fold}.csv', smiles_emb=smiles_emb,
                                      target_emb=target_emb, smiles_idx=smiles_idx, smiles_graph=smiles_graph,
                                      target_graph=target_graph, smiles_len=smiles_len, target_len=target_len)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        train(model, train_loader, optimizer, epoch)
        G, P = predicting(model, test_loader)
        val1 = get_mse(G, P)
        if val1 < best_mse:
            best_mse = val1
            best_epoch = epoch + 1
            if model_file_name is not None:
                torch.save(model.state_dict(), model_file_name)
            log(f'mse improved at epoch {best_epoch}, best_mse {best_mse}')
        else:
            log(f'current mse: {val1} , No improvement since epoch {best_epoch}, best_mse {best_mse}')

        schedule.step(val1)
