import torch

def read_data_and_save_to_pt(file_path, output_file):
    # 初始化一個列表來保存所有數據
    all_data = []
    
    # 讀取文件
    with open(file_path, 'r') as f:
        for line in f:
            # 將每行轉換為整數列表
            numbers = list(map(int, line.strip().split(',')))
            # 添加到all_data列表中
            all_data.append(numbers)
    
    # 將列表轉換為PyTorch張量
    data_tensor = torch.tensor(all_data, dtype=torch.int)
    
    # 保存張量到.pt檔
    torch.save(data_tensor, output_file)
    
    print(f'Data saved to {output_file}. Shape: {data_tensor.shape}')

# 假設你的文件路徑和想要保存的.pt檔名
file_path = '/data/yakowei/research/dataset/label/pubchem_ring/pubchem_re/2m_chemBL_pubchem_10_40p.txt'
output_file = '/data/yakowei/research/dataset/label/pubchem_ring/2m_chemBL_Label_10_40p.pt'
#/data/yakowei/research/dataset/label/pubchem_ring/2m_chemBL_Label_20_40pr.txt

# 調用函數
read_data_and_save_to_pt(file_path, output_file)