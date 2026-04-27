import torch

def read_data_and_convert_to_tensor(file_path):
    tensors = []
    with open(file_path, 'r') as file:
        for line in file:
            # 分割每行數據，並去除開頭的molecule_編號
            data = line.strip().split('\t')[1:]
            # 將數據轉換為浮點數，然後轉換為tensor
            tensor = torch.tensor([float(x) for x in data])
            tensors.append(tensor)
    return tensors

# 讀取數據並轉換
file_path = '/data/yakowei/research/dataset/label/checkmol/2m_chemBL_Label_10_19c.txt'
tensors = read_data_and_convert_to_tensor(file_path)
tensors=torch.stack(tensors)
# 將結果儲存為.pt檔
torch.save(tensors, '/data/yakowei/research/dataset/label/checkmol/2m_chemBL_Label_10_19c.pt')

print("數據已轉換並儲存為test_Label.pt")