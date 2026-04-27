#python3 -m research.preprocessing.preprocessing_matrix 
import numpy as np
from research.encode import smile
import os
import torch
from torchvision.transforms.functional import to_tensor
def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return lines

result = read_txt_file("/data/yakowei/research/dataset/image/2m_chemBL_smile_10_40.txt")
def preprocess(graph_matrix):
    graph_matrix = to_tensor(graph_matrix)
    return graph_matrix

x_0_batch = []
for line in result:
    graph_matrix = smile.smile2graph(line)
    #print("graphshape", graph_matrix.shape)
    processed_matrix = preprocess(graph_matrix)
    #print("processed_matrix", processed_matrix.shape)
    x_0_batch.append(processed_matrix)

# 将所有处理后的 GraphMatrix 堆叠起来
x_0_batch = torch.stack(x_0_batch)
torch.save(x_0_batch, '/data/yakowei/research/dataset/image/2m_chemBL_smile_10_40.pt')