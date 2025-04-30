import os
import torch
from torch.utils.data import Dataset, DataLoader
from Bio.PDB import PDBParser
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import  pandas as pd
class ProteinEGNNDataset(Dataset):
    def __init__(self, pdb_folder, seq_len=1000, edge_threshold=8.0):
        self.pdb_folder = pdb_folder
        self.pdb_files = [f for f in os.listdir(pdb_folder) if f.endswith('.pdb')]
        self.seq_len = seq_len
        self.edge_threshold = edge_threshold
        self.parser = PDBParser(QUIET=True)

    def __len__(self):
        return len(self.pdb_files)

    def __getitem__(self, idx):
        pdb_path = os.path.join(self.pdb_folder, self.pdb_files[idx])
        structure = self.parser.get_structure('protein', pdb_path)

        coords, feats = [], []

        for model in structure:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        atom = residue['CA']
                        coords.append(atom.coord)
                        feats.append(self.residue_to_feature(residue))
        coords = torch.tensor(coords, dtype=torch.float)
        feats = torch.stack(feats)

        # padding or truncate
        coords, feats = self.pad_or_crop(coords, feats, self.seq_len)

        # compute edges
        edge_attr = self.get_edge_attr(coords, self.edge_threshold)

        return feats.unsqueeze(0), coords.unsqueeze(0), edge_attr.unsqueeze(0)  # shape: (1, N, ...)

    def residue_to_feature(self, residue):
        amino_acids = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
                       'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','UNK']
        resname = residue.get_resname()
        one_hot = torch.tensor([resname == aa for aa in amino_acids], dtype=torch.float)
        return one_hot

    def pad_or_crop(self, coords, feats, target_len):
        N = coords.shape[0]
        if N >= target_len:
            return coords[:target_len], feats[:target_len]
        else:
            pad_len = target_len - N
            pad_coords = torch.zeros((pad_len, 3))
            pad_feats = torch.zeros((pad_len, feats.shape[1]))
            return torch.cat([coords, pad_coords]), torch.cat([feats, pad_feats])

    def get_edge_attr(self, coords, threshold):
        N = coords.shape[0]
        edges = torch.zeros((N, N, 1))
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                dist = torch.norm(coords[i] - coords[j])
                if dist < threshold:
                    edges[i, j, 0] = dist / threshold  # normalized distance
        return edges
def pdb_to_pyg():
    import os
    import numpy as np
    import torch
    from Bio.PDB import PDBParser
    from tqdm import tqdm

    AA_LIST = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
               'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'UNK']
    aa2idx = {aa: i for i, aa in enumerate(AA_LIST)}

    def residue_one_hot(resname):
        vec = np.zeros(len(AA_LIST))
        vec[aa2idx.get(resname, aa2idx['UNK'])] = 1
        return vec

    def pdb_to_pyg(pdb_file, distance_threshold=8.0):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file)

        coords, features = [], []
        for model in structure:
            for chain in model:
                for res in chain:
                    if 'CA' in res:
                        ca = res['CA'].get_coord()
                        coords.append(ca)
                        features.append(residue_one_hot(res.get_resname()))

        coords = np.array(coords)  # (N, 3)
        features = np.array(features)  # (N, F)

        N = coords.shape[0]
        # if N < SEQ_LEN:
        #     padded = np.zeros((SEQ_LEN, 3), dtype=np.float32)
        #     padded[:N] = coords
        #     coords = padded
        #     mask = torch.zeros(SEQ_LEN, dtype=torch.bool)
        #     mask[:N] = 1
        # else:
        #     coords = coords[:SEQ_LEN]
        #     mask = torch.ones(SEQ_LEN, dtype=torch.bool)

        # edge_feats = np.zeros((N, N, 4))  # 1D edge feature: pairwise distance
        edge_index = []
        edge_attr = []
        for i in range(N):
            for j in range(N):
                if i != j:
                    vec = coords[i] - coords[j]
                    dist = np.linalg.norm(vec)
                    if dist <= distance_threshold:
                        edge_index.append([i, j])
                        edge_attr.append([
                            dist / distance_threshold,  # 距离归一化
                            vec[0] / (dist + 1e-6),  # 方向 x
                            vec[1] / (dist + 1e-6),  # 方向 y
                            dist ** 2  # 距离平方
                        ])



        return torch.tensor(features, dtype=torch.float32), \
               torch.tensor(coords, dtype=torch.float32), \
               torch.tensor(edge_index, dtype=torch.long).T, \
               torch.tensor(edge_attr, dtype=torch.float32)

    def process_pdb_folder(input_folder, output_folder, distance_threshold=8.0):
        # Ensure output directory exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # List all PDB files in the input folder
        pdb_files = [f for f in os.listdir(input_folder) if f.endswith('.pdb')]

        for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
            pdb_path = os.path.join(input_folder, pdb_file)

            # Generate features, coordinates, and edge features
            features, coords, edge_index, edge_attr= pdb_to_pyg(pdb_path, distance_threshold)

            # Create a PyTorch Geometric Data object
            data = DATA(x=features, pos=coords, edge_index=edge_index, edge_attr=edge_attr)

            # Save the Data object to the output folder
            output_path = os.path.join(output_folder, f"{os.path.splitext(pdb_file)[0]}.pt")
            torch.save(data, output_path)

    input_folder = 'kiba/pdb'
    output_folder = 'kiba/pyg'

    process_pdb_folder(input_folder, output_folder, distance_threshold=8.0)
def sm_to_sdf():
    # 读取 DataFrame
    df = pd.read_csv('kiba/kiba_processed.csv')  # 假设数据是从 CSV 读取的，修改为实际路径

    df = df['compound_iso_smiles'].unique()
    # print(df)
    # 创建一个保存 SDF 文件的目录
    output_dir = 'kiba/sdf/'
    print(len(df))
    sm_id = pd.DataFrame(columns=['smiles','id'])
    for i in range(len(df)):
        row = df[i]
        sm_id.loc[i] = [row,i]
    sm_id.to_csv('kiba/sm_id.csv',index=False)

        

    # # 遍历 DataFrame 的 compound_iso_smiles 列
    # for idx in range(len(df)):
    #     smiles = df[idx]

    #     mol = Chem.MolFromSmiles(smiles)
    #     if mol is None:
    #         print(f"警告: 无法从 SMILES '{smiles}' 生成分子，跳过该分子.")
    #         continue  # 如果 SMILES 无法解析，跳过当前分子

    #     mol = Chem.AddHs(mol)
    #     status = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    #     if status != 0:
    #         print(f"警告: 无法为分子 {smiles} 生成有效的三维结构，跳过该分子.")
    #         continue  # 如果三维结构生成失败，跳过当前分子
    #     AllChem.UFFOptimizeMolecule(mol)

    #     # 获取通用分子名（可以从分子的属性中获取）
    #     mol_name = smiles

    #     sdf_filename = os.path.join(output_dir, f"{mol_name}.sdf")
    #     with Chem.SDWriter(sdf_filename) as writer:
    #         writer.write(mol)  # 将分子写入 SDF 文件
def sdf_to_pyg():
    import os
    import torch
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from torch_geometric.data import Data
    import numpy as np
    # 设置文件夹路径
    input_dir = 'kiba/sdf/'  # SDF 文件夹路径
    output_dir = 'kiba/pyg/'  # 输出目录

    # 定义常见原子类型
    COMMON_ATOMIC_TYPES = [
        'H', 'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I',
        'He', 'Li', 'Be', 'B', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca',
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn'
    ]
    atomic_type_to_index = {atom: idx for idx, atom in enumerate(COMMON_ATOMIC_TYPES)}
    
    def one_hot(index, length):
        vec = torch.zeros(length)
        if 0 <= index < length:
            vec[index] = 1.0
        return vec

    def molecule_to_pyg_graph(mol):
        # 获取原子数目
        mol = Chem.RemoveHs(mol)
        num_atoms = mol.GetNumAtoms()

        # 使用常见原子类型映射获取原子特征
        atom_features = []
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            atom_symbol = Chem.GetPeriodicTable().GetElementSymbol(atomic_num)
            index = atomic_type_to_index.get(atom_symbol, len(COMMON_ATOMIC_TYPES))  # 使用索引或默认值处理未知类型
            atom_features.append(one_hot(index, len(COMMON_ATOMIC_TYPES) + 1))  # +1 用于未知类型

        # 将 atom_features 转换为张量
        atom_features = torch.stack(atom_features)  # 使用 stack 而不是 tensor
        atom_features = atom_features.view(-1, len(COMMON_ATOMIC_TYPES) + 1)  # 原子特征

        # 获取三维坐标
        conf = mol.GetConformer()
        positions = []
        for atom_id in range(num_atoms):
            pos = conf.GetAtomPosition(atom_id)
            positions.append([pos.x, pos.y, pos.z])

        # 创建一个PyG图结构
        positions = torch.tensor(positions, dtype=torch.float)  # 原子坐标

        # 生成边 (这里简单地假设所有原子都是连通的)
        edge_index = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # 返回PyG数据对象
        return Data(x=atom_features, edge_index=edge_index, pos=positions)



    # 遍历文件夹中的所有 SDF 文件
    for sdf_file in os.listdir(input_dir):
        if sdf_file.endswith(".sdf"):
            sdf_path = os.path.join(input_dir, sdf_file)
            base_filename = os.path.splitext(sdf_file)[0]
            # 读取SDF文件
            supplier = Chem.SDMolSupplier(sdf_path)

            for idx, mol in enumerate(supplier):
                if mol is None:
                    continue

                # 转换为PyG图数据
                pyg_graph = molecule_to_pyg_graph(mol)

                # 保存PyG数据
                torch.save(pyg_graph, os.path.join(output_dir, f"{base_filename}.pt"))
                print(f"保存: {base_filename}.pt")


# 用法示例
if __name__ == '__main__':
    # dataset = ProteinEGNNDataset('./pdb', seq_len=1000, edge_threshold=8.0)
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    sdf_to_pyg()
    # sm_to_sdf()

    pdb_to_pyg()
