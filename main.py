import numpy as np
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
import gc
from dataset import DTA_Dataset
from sklearn.model_selection import KFold
from model import *
from torch import nn as nn
from moe import *
from egnn import EGNN
from mamba_ssm import Mamba
from sklearn.metrics import r2_score
from lifelines.utils import concordance_index
#############################################################################

CUDA = '0'
device = torch.device('cuda:' + CUDA)
LR = 1e-3
NUM_EPOCHS = 100
seed = 0
batch_size = 128
dataset_name = 'davis'


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
        self.sm_init = nn.Linear(embedding_dim, lstm_dim)
        self.smiles_lstm = nn.LSTM(lstm_dim, lstm_dim, self.bilstm_layers, batch_first=True,
                                   bidirectional=True, dropout=dropout_rate)
        self.smile_mamba = Mamba(d_model=embedding_dim, d_state=16, d_conv =4, expand =2)
        self.sm_fc = nn.Linear(lstm_dim, embedding_dim)
        self.smiles_conv = TCNBlock(embedding_dim)
        self.enhance1 = SpatialGroupEnhance_for_1D(groups=20)

        # 蛋白质相关
        self.protein_vocab = protein_vocab
        self.protein_embed = nn.Embedding(protein_vocab + 1, embedding_dim, padding_idx=0)
        self.pr_init = nn.Linear(embedding_dim, lstm_dim)
        self.protein_lstm = nn.LSTM(lstm_dim, lstm_dim, self.bilstm_layers, batch_first=True,
                                    bidirectional=True, dropout=dropout_rate)
        self.protein_mamba = Mamba(d_model=embedding_dim, d_state=16, d_conv =4, expand =2)
        self.pr_fc = nn.Linear(lstm_dim, embedding_dim)
        self.protein_conv = TCNBlock(embedding_dim)
        self.enhance2 = SpatialGroupEnhance_for_1D(groups=200)

        # 输出层
        # Point-wise Feed Forward Network
        self.pwff_1 = nn.Linear(embedding_dim * 3, embedding_dim * 4)
        self.pwff_2 = nn.Linear(embedding_dim * 4, embedding_dim * 3)
        self.out_fc1 = nn.Linear(embedding_dim * 3, 256 * 8)
        self.out_fc2 = nn.Linear(256 * 8, hidden_dim)
        self.out_fc3 = nn.Linear(hidden_dim, 1)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # 其他网络组件
        self.norm = RMSNorm(embedding_dim)
        self.sgemb = nn.Linear(31, embedding_dim)
        self.tgemb = nn.Linear(21, embedding_dim)
        self.MGNN = MultiLayerGCN(num_features=embedding_dim, hidden_dim=embedding_dim//2, num_classes=embedding_dim, num_layers=3)
        self.M_GNN = MultiLayerGCN(num_features=embedding_dim+1, hidden_dim=embedding_dim//2, num_classes=embedding_dim, num_layers=3)
        self.out_2g_fc1 = nn.Linear(hidden_dim*2, 256 * 8)
        self.out_2g_fc2 = nn.Linear(256 * 8, hidden_dim)

        self.tEGNN = EGNN(21,128,embedding_dim//2)
        self.sEGNN = EGNN(31,128,embedding_dim//2)
        self.out_3g_fc1 = nn.Linear(hidden_dim, 256 * 8)
        self.out_3g_fc2 = nn.Linear(256 * 8, hidden_dim)
        # self.crossattn = CrossAttentionMultiHead(hidden_dim, 2)
        self.lmf = LMF(dim=hidden_dim, rank=4, dropout=0.1)
        # self.w = nn.Parameter(torch.tensor(0.5))

        self.fusion_0 = nn.Linear(embedding_dim*2, embedding_dim)
        self.fusion_1 = nn.Linear(embedding_dim, embedding_dim *4)
        self.fusion_2 = nn.Linear(embedding_dim *4, embedding_dim)
        self.ln= nn.LayerNorm(embedding_dim)

    def forward(self, data, reset=False):
        """
        前向传播方法。

        :param data: 输入数据
        :param reset: 是否重置状态
        :return: 模型输出和标签
        """
        
        s_data, t_data = data
        batch_size = len(t_data)
        smiles = t_data.smiles.to(device)
        protein = t_data.protein.to(device)
        smiles = smiles.view(batch_size,540)
        protein = protein.view(batch_size,1000)
        smiles_lengths = t_data.smiles_lengths
        protein_lengths = t_data.protein_lengths

        # SMILES 处理
        smiles = self.smiles_embed(smiles)
        smiles = self.sm_init(smiles)
        # smiles = self.smiles_conv(smiles)
        smiles = self.enhance1(smiles)
        smiles,_ = self.smiles_lstm(smiles)
        smiles_mamba_out = self.smile_mamba(smiles)
        smiles = smiles + smiles_mamba_out
        smiles = self.layer_norm(smiles)

        # 蛋白质处理
        protein = self.protein_embed(protein)
        protein = self.pr_init(protein)
        # protein = self.protein_conv(protein)
        protein = self.enhance2(protein)
        protein,_ = self.protein_lstm(protein)
        protein_mamba_out = self.protein_mamba(protein)
        protein = protein + protein_mamba_out
        protein = self.layer_norm(protein)

        out = torch.cat([smiles,protein],dim=1)
        if reset:
            return smiles, protein

        # 生成掩码
        smiles_mask = self.generate_masks(smiles, smiles_lengths, self.n_heads)
        protein_mask = self.generate_masks(protein, protein_lengths, self.n_heads)
        out_mask = torch.cat([smiles_mask,protein_mask],dim=1)
        # 池化
        smiles_out = self.masked_mean_pooling(smiles, smiles_mask)
        protein_out = self.masked_mean_pooling(protein, protein_mask)
        out = self.masked_mean_pooling(out,out_mask)
        out = torch.cat([smiles_out, protein_out,out], dim=-1)
        # Point-wise Feed Forward Network
        pwff = self.dropout(self.relu(self.pwff_1(out)))
        pwff = self.dropout(self.relu(self.pwff_2(pwff)))
        out = pwff + out
        
        out = self.dropout(self.relu(self.out_fc1(out)))
        out = self.dropout(self.relu(self.out_fc2(out)))
        # out = self.bn(out)
        
        s3g = self.sEGNN(s_data.x,s_data.pos,s_data.edge_index,s_data.batch)
        t3g = self.tEGNN(t_data.x,t_data.pos,t_data.edge_index,t_data.batch)
        out3g = torch.cat([s3g,t3g],dim=-1)
        out3g = self.ln(out3g)
        pwff = self.dropout(self.relu(self.out_3g_fc1(out3g)))
        pwff = self.dropout(self.relu(self.out_3g_fc2(pwff)))
        out3g = pwff + out3g

        s_data.x = self.sgemb(s_data.x)
        t_data.x = self.tgemb(t_data.x)
        s2g = self.MGNN(s_data)
        t2g = self.MGNN(t_data)
        # all2g = self.M_GNN(data)
        out2g = torch.cat([s2g,t2g],dim=-1)
        pwff = self.dropout(self.relu(self.out_2g_fc1(out2g)))
        out2g = self.dropout(self.relu(self.out_2g_fc2(pwff)))
        out2g = self.ln(out2g)

        # outg = self.w * out2g + (1 - self.w) * out3g
        # # out_ = out
        # out_ = self.crossattn(outg,out,out)
        out_ = self.lmf(out,out2g,out3g)
        # out_ = torch.cat([out,outg],dim=-1)
        # out_ = self.fusion_0(out_)
        # out_ = self.ln(out_)
        out_ = self.fusion_1(out_)
        out_ = self.fusion_2(out_)
        out = out_ + out
        return self.out_fc3(out).squeeze(), t_data.y

    def generate_masks(self, adj, adj_sizes, n_heads):
        """
        生成掩码。

        :param adj: 输入张量
        :param adj_sizes: 大小
        :param n_heads: 头数
        :return: 掩码张量
        """
        out = torch.ones(adj.shape[0], adj.shape[1])
        max_size = adj.shape[1]
        if isinstance(adj_sizes, int):
            out[0, adj_sizes:max_size] = 0
        else:
            for e_id, drug_len in enumerate(adj_sizes):
                out[e_id, drug_len:max_size] = 0
        return out.cuda(device=adj.device)

    def masked_mean_pooling(self, x, mask):
        """
        掩码平均池化。

        :param x: 输入张量
        :param mask: 掩码
        :return: 池化后的张量
        """
        mask = mask.unsqueeze(-1)  # [B, L, 1]
        x = x * mask  # zero out padded positions
        sum_x = x.sum(dim=1)  # [B, D]
        lengths = mask.sum(dim=1).clamp(min=1e-6)  # [B, 1]
        return sum_x / lengths  # [B, D]

#############################################################################

df = pd.read_csv(f'./{dataset_name}/{dataset_name}_processed.csv')
smiles = set(df['compound_iso_smiles'])
target = set(df['target_key'])
target_seq = {}
for i in range(len(df)):
    target_seq[df.loc[i, 'target_key']] = df.loc[i, 'target_sequence']


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



data_set = f"{dataset_name}_processed"
mode = ['default', 'drug_cold', 'target_cold', 'all_cold']
mode_name = mode[0]
seed = [18,283,839,12,74]
# model_file_name = './Model/' + dataset_name + '_' + mode_name + '.pt'
model_file_name = './Model/' + dataset_name + '_LMF.pt'
dim = 128
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "Total": total_params/1e6,
        "Trainable": trainable_params/1e6
    }
for fold in range(5):
    print("Building model...")
    model = DMFF(embedding_dim=dim * 2, lstm_dim=dim, hidden_dim=dim *2, dropout_rate=0.2,
                 alpha=0.2, n_heads=8, bilstm_layers=2, protein_vocab=26,
                 smile_vocab=45, theta=0.5).to(device)
    param_stats = count_parameters(model)
    print(f"Total Params: {param_stats['Total']:,}")
    print(f"Trainable Params: {param_stats['Trainable']:,}")
    # load model
    if os.path.exists(model_file_name):
        save_model = torch.load(model_file_name)
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=2e-5,verbose=True)

    best_mse = 1000
    best_test_mse = 1000
    best_epoch = -1
    best_test_epoch = -1
    print(f"Fold {fold + 1}")
    log(f'train on {dataset_name}_{mode_name}')
    for epoch in range(NUM_EPOCHS):
        print("No {} epoch".format(epoch))
        if epoch == 0:
            train_dataset = DTA_Dataset(root='./', path=f'./{dataset_name}/{mode_name}/'+ 'train.csv', smiles_emb=smiles_emb, target_emb=target_emb, smiles_len=smiles_len, target_len=target_len,mode= mode_name)
            val_dataset = DTA_Dataset(root='./', path=f'./{dataset_name}/{mode_name}/'+ 'valid.csv', smiles_emb=smiles_emb, target_emb=target_emb, smiles_len=smiles_len, target_len=target_len,mode= mode_name)
            test_dataset = DTA_Dataset(root='./', path=f'./{dataset_name}/{mode_name}/'+ 'test.csv', smiles_emb=smiles_emb, target_emb=target_emb, smiles_len=smiles_len, target_len=target_len,mode= mode_name)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        train(model, train_loader, optimizer, epoch)
        G, P = predicting(model, val_loader)
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

        if epoch % 10 == 0:
            G, P = predicting(model, test_loader)
            cindex, rm2, mse = calculate_metrics_and_return(G, P, test_loader)
            # mse = get_mse(G, P)
            ci = concordance_index(G, P)
            # r2 = r2_score(G, P)
            log(f'epoch {epoch} test mse:{mse}, r2:{rm2}, ci:{ci}')
            file_name = './Model/' + dataset_name + '_' + str(epoch) + '.pt'
            if file_name is not None:
                torch.save(model.state_dict(), file_name)
            print(f"epoch {epoch}:mse {mse} cindex {cindex} rm2 {rm2}")

    save_model = torch.load(model_file_name)
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    G, P = predicting(model, test_loader)
    cindex, rm2, mse = calculate_metrics_and_return(G, P, test_loader)
    break
