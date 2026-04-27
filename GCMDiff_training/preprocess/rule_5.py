from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors as rdescriptors
import torch

# 讀取 SMILES 文件並處理每一行
def process_smiles(file_path):
    results = []
    with open(file_path, 'r') as file:
        for line in file:
            smiles = line.strip()
            mol = Chem.MolFromSmiles(smiles)
            if mol:  # 確保 SMILES 是有效的
                results.append([
                    int(Descriptors.MolWt(mol) <= 500),
                    int(Descriptors.MolLogP(mol) <= 5),
                    int(rdescriptors.CalcNumLipinskiHBA(mol) <= 10),
                    int(rdescriptors.CalcNumLipinskiHBD(mol) <= 5)
                ])
    return results

# 假設你的 SMILES 數據存儲在 'data.txt' 檔案中
file_path = '/data/yakowei/research/dataset/test.txt'
smiles_data = process_smiles(file_path)

# 將結果轉換成 tensor
results_tensor = torch.tensor(smiles_data, dtype=torch.int8)

# 檢查 tensor 的大小
print(results_tensor.size())  # 應該輸出 torch.Size([2119703, 4])

# 保存 tensor 到 .pt 檔案
torch.save(results_tensor, './test.pt')
