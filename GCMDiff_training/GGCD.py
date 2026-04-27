import torch
from torch.utils.data import Dataset,DataLoader
from denoising_diffusion.classifier_free_guidance_v4 import Unet, GaussianDiffusion
import tqdm 

#CUDA_VISIBLE_DEVICES=1 python3 GGCD.py 

class CustomDataset(Dataset):
    def __init__(self, smiles, labels):
        self.smiles = smiles.to('cuda')
        self.labels = labels.to('cuda')
    
    def __len__(self):
        """返回数据集中的数据数"""
        return len(self.smiles)
    
    def __getitem__(self, idx):
        """
        获取一个样本
        :param idx: 样本的索引值
        """
        smile = self.smiles[idx]
        original_label = self.labels[idx]  
        return smile, original_label

model = Unet(
    dim = 64,
    num_classes=524,
    dim_mults = (1, 2, 4, 8),
    cond_drop_prob = 0.2
).to('cuda')

diffusion = GaussianDiffusion(
    model,
    image_size = 40,
    timesteps = 1000
).to('cuda')
# state_dict = torch.load('/data/yakowei/research/Final_Output/model_weight/64_1000_condi/0609/100_0.000943687919061631.pt', map_location="cuda")
# # 然後使用 load_state_dict() 並設定 strict=False
# model.load_state_dict(state_dict, strict=False)

smiles=torch.load("./dataset/img/example_img.pt").to("cuda", dtype=torch.float)
labels=torch.load("./dataset/label/example_label.pt").to("cuda", dtype=torch.float)

# 創建 Dataset
dataset = CustomDataset(smiles, labels)
# 使用 DataLoader
#batch_size suggest 128 (2 is for test)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True,drop_last=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999))

pbar = tqdm.tqdm(range(10000))
for epoch in pbar:
#    scheduler.step()
    for sub_batch in dataloader:
        sub_smiles,sub_label=sub_batch
        optimizer.zero_grad()
        loss = diffusion(sub_smiles,classes =sub_label)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({'Loss': loss.item()})
        record=loss.item()
    if  (epoch+1)%10==0:   
        torch.save(model.state_dict(), f'./denoising_diffusion/model_weight/{epoch+1}_{record}.pt')
    