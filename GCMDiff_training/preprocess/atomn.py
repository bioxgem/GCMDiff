from rdkit import Chem
import torch

# 假設你的 SMILES 數據存儲在 'data.txt' 檔案中
file_path = '/data/yakowei/research/dataset/image/2m_chemBL_smile_10_40.txt'

# 初始化一個列表來存儲原子數
atom_counts = []

# 讀取文件並計算每個 SMILES 的原子數
with open(file_path, 'r') as file:
    for line in file:
        smiles = line.strip()
        mol = Chem.MolFromSmiles(smiles)
        if mol:  # 確保 SMILES 是有效的
            atom_counts.append([mol.GetNumAtoms()])  # 使用列表以便於後續轉換成 tensor

# 將原子數列表轉換成 tensor
atom_counts_tensor = torch.tensor(atom_counts)

# 檢查 tensor 的大小
print(atom_counts_tensor.size())  # 應該輸出 torch.Size([2119703, 1])

# 保存 tensor 到 .pt 檔案
torch.save(atom_counts_tensor, './atom_counts.pt')
