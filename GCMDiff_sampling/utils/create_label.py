import numpy as np
import torch


def generate_random_atomn(data):
    # 提取原子总数和数量
    atom_totals = list(data.keys())
    quantities = list(data.values())

    # 计算概率
    total_quantity = sum(quantities)
    probabilities = [q / total_quantity for q in quantities]

    # 使用这些概率随机选择一个原子总数
    random_atom_total = np.random.choice(atom_totals, p=probabilities)
    return [random_atom_total]


# extend_bool=False
# batch_sample=4

def create_label_random_atomn(rule_5,checkmol,checkmol_num,pubchem,pubchem_num,ring,ring_num,batch_sample):
    data = {
    10: 4039, 11: 6312, 12: 9253, 13: 12331, 14: 17357,
    15: 24861, 16: 33216, 17: 43511, 18: 56767, 19: 69567,
    20: 84955, 21: 92994, 22: 105566, 23: 114568, 24: 123769,
    25: 126328, 26: 124923, 27: 117942, 28: 116211, 29: 110983,
    30: 106975, 31: 99812, 32: 92224, 33: 82494, 34: 74274,
    35: 63876, 36: 55155, 37: 46511, 38: 40287, 39: 33500,
    40: 29142
    }
    batch_tensors=[]
    for i in range(batch_sample):     
        atomn=generate_random_atomn(data)
        numpy_array = label_np(atomn,rule_5,checkmol,checkmol_num,pubchem,pubchem_num,ring,ring_num)
        # print(numpy_array.shape)
        numpy_array=np.squeeze(numpy_array)
        smile_class = torch.tensor(numpy_array).to("cuda", dtype=torch.float)
        batch_tensors.append(smile_class)
    smile_classes = torch.stack(batch_tensors)
    return smile_classes

def generate_array_with_counts(indices, counts,length):
    adjusted_indices = [index - 1 for index in indices]
    array = np.zeros((1,length))
    for index, count in zip(adjusted_indices, counts):
        array[0, index] = count
    return array

def generate_array_with_list(condi_list):
    list_array=np.array(condi_list)
    list_array=list_array.reshape(1,len(condi_list))
    return list_array

def label_np(atomn,rule_5,checkmol,checkmol_num,pubchem,pubchem_num,ring,ring_num):
    n=generate_array_with_list(atomn)
    r5=generate_array_with_list(rule_5)
    c=generate_array_with_counts(checkmol,checkmol_num,204)
    p=generate_array_with_counts(pubchem,pubchem_num,168)
    r=generate_array_with_counts(ring,ring_num,147)
    result = np.concatenate((n,r5,c, p, r),axis=1)
    return result

def create_label_np(atomn,rule_5,checkmol,checkmol_num,pubchem,pubchem_num,ring,ring_num,batch_sample,extend_bool):
    numpy_array = label_np(atomn,rule_5,checkmol,checkmol_num,pubchem,pubchem_num,ring,ring_num)

    return numpy_array

def create_label(atomn,rule_5,checkmol,checkmol_num,pubchem,pubchem_num,ring,ring_num,batch_sample,extend_bool):
    numpy_array = label_np(atomn,rule_5,checkmol,checkmol_num,pubchem,pubchem_num,ring,ring_num)

    smile_classes = torch.tensor(numpy_array).to("cuda", dtype=torch.float)
    smile_classes=smile_classes.repeat(batch_sample,1)
    return smile_classes

def creat_label_extend_randon(random_np,extend_bool):
    batch_tensors=[]
    for row in random_np:
        numpy_array=row
        smile_class = torch.tensor(numpy_array).to("cuda", dtype=torch.float)
        batch_tensors.append(smile_class)
    smile_classes = torch.stack(batch_tensors)
    return smile_classes

def create_moiety_list(checkmol,pubchem,ring):
    checkmol=[x-1 for x in checkmol]
    last_value_list1 = 204
    pubchem=[x-1+last_value_list1 for x in pubchem]
    last_value_list2 = 168+204
    ring=[x-1+last_value_list2 for x in ring]
    merged_list = sorted(checkmol+pubchem+ring)
    return merged_list

# a=create_label(atomn,rule_5,checkmol,checkmol_num,pubchem,pubchem_num,ring,ring_num,batch_sample,extend_bool)
# print(a.shape)
if __name__=="__main__":
    atomn=[20]
    rule_5=[1,1,1,1]
    checkmol=[3, 5, 37, 38, 75, 78, 79, 201, 202] 
    checkmol_num= [1, 1, 1, 1, 3, 3, 1, 1, 3]
    pubchem=[64, 65, 85, 86, 106, 107, 112] 
    pubchem_num= [4, 2, 7, 3, 7, 3, 1]
    ring=[5, 6, 25, 46, 108] 
    ring_num= [13, 2, 2, 2, 1]
    batch_sample=2
    a=create_label_random_atomn(rule_5,checkmol,checkmol_num,pubchem,pubchem_num,ring,ring_num,batch_sample)
    