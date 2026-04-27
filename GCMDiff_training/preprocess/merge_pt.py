import torch

# 假设文件已经被加载到内存
tensor1 = torch.load('/data/yakowei/research/dataset/label/atom_number/atom_counts.pt')  # torch.Size([2119703, 1])
tensor2 = torch.load('/data/yakowei/research/dataset/label/rule_5/rule_5.pt')  # torch.Size([2119703, 4])
tensor3 = torch.load('/data/yakowei/research/dataset/label/checkmol/2m_chemBL_Label_10_40c.pt')  # torch.Size([2119703, 204])
tensor4 = torch.load('/data/yakowei/research/dataset/label/pubchem_ring/2m_chemBL_Label_10_40pr.pt')  # torch.Size([2119703, 315])

# 合并 Tensor
merged_tensor = torch.cat((tensor1, tensor2, tensor3, tensor4), dim=1)

# 保存合并后的 Tensor 到文件
torch.save(merged_tensor, './2m_chemBL_Label_10_40.pt')

# 验证文件是否保存成功
print(f'Merged tensor saved: {merged_tensor.size()}')
print(f'Data type of merged tensor: {merged_tensor.dtype}')