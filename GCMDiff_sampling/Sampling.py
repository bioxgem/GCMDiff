import os
import torch
import faulthandler
from denoising_diffusion.classifier_free_guidance_v4 import Unet, GaussianDiffusion
from utils.tools import revearseimg,delete_files_in_folder
from utils.create_label import create_label
from utils.molgenerate import mol_generate,rdkit_draw

#usage :
# set environment: conda activate protein_design (*** enter protein_design ***)
# return base: conda deactivate  (***return base environemnt ***)
# check GPU usage: nvidia-smi
# CUDA_VISIBLE_DEVICES=1 python3 Sampling.py   // execute program 

# help to debug 帮助诊断因底层问题
faulthandler.enable()

model = Unet(
    dim = 64,
    num_classes=524,
    dim_mults = (1, 2, 4, 8),
    cond_drop_prob = 0
).to('cuda')
# propram in the direction: denoising_diffusion
diffusion = GaussianDiffusion(
    model,
    image_size = 40,
    timesteps = 1000
).to('cuda')

# load pre-trained model 
model.load_state_dict(torch.load('./denoising_diffusion/model_weight/150_0.0008395609329454601.pt',map_location="cuda"))

# set generated compound condition 
#原子數量 10~40
atomn=[15]
#rule of five, 1 or 0
rule_5=[1,1,1,1]

# compound moiety link: http://140.113.120.248/BioXGEM-drug/Compound/compound_moiety_base_features.php
#checkmol index
checkmol=[201]
#pubchem index 
pubchem=[] 
#ring in drug index
ring=[1]
#生成數量
batch_sample=3


#test!!!
checkmol_num=[1] * len(checkmol)
pubchem_num= [1] * len(pubchem)
ring_num= [1] * len(ring)
extend_bool=False

#組成條件
smile_classes=create_label(atomn,rule_5,checkmol,checkmol_num,pubchem,pubchem_num,ring,ring_num,batch_sample,extend_bool)

#生成mol檔位置
mol_path='./output/mol_file'
#清空之前的內容
delete_files_in_folder(mol_path)
#生成img檔位置
img_path='./output/img'
#清空之前的內容
delete_files_in_folder(img_path)

sampled_matrix= diffusion.sample(
    classes = smile_classes,
    #控制條件參數 0=無條件影響 ，建議3~5之間
    cond_scale = 4. ,
    apply_mask=False ,
)


smile_matrix=sampled_matrix.detach().cpu().numpy()
for i in range(batch_sample):
    sub_batch=smile_matrix[i,:,:,:]
    smile_img = sub_batch.transpose(1, 2, 0)
    transposed_smile_img=revearseimg(smile_img)
#存mol檔
    output_mol=f'./output/mol_file/img_{i}.mol'
    mol_generate(transposed_smile_img,output_mol)
#rdkit讀mol檔作圖
    rd_img=f'./output/img/img_ns_{i}.png'
    rdkit_draw(output_mol,rd_img,sanitize=False) 
    #sanitize=True 錯誤的mol檔不會生成img
