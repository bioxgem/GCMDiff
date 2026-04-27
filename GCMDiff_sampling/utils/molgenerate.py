from openbabel import openbabel
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from rdkit.Chem import AllChem
import os
base_path = os.path.dirname(os.path.abspath(__file__))
mol_file_path = os.path.join(base_path, "tmp/tmp.mol")
def roundmatrix(matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            for k in range(matrix.shape[2]):
                if matrix[i][j][k]>0:   
                    matrix[i][j][k]=round(matrix[i][j][k])
                else :
                    matrix[i][j][k]=0
    return matrix

def roundmatrix_cut(matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            for k in range(matrix.shape[2]):
                if abs(matrix[i][j][k]-round(matrix[i][j][k]))<0.2:
                    matrix[i][j][k]=round(matrix[i][j][k])
                else:
                    matrix[i][j][k]=0
    return matrix

def element_number_to_atomic_number(element_number):
    # 定义元素和它们的原子序数
    heavyatom_atomic_numbers = {
        1: 34,  # Se
        2: 3,   # Li
        3: 9,   # F
        4: 14,  # Si
        5: 52,  # Te
        6: 8,   # O
        7: 5,   # B
        8: 15,  # P
        9: 7,   # N
        10: 33, # As
        11: 53, # I
        12: 35, # Br
        13: 19, # K
        14: 6,  # C
        15: 20, # Ca
        16: 11, # Na
        17: 16, # S
        18: 17, # Cl
        19: 1,  # H
        20: 85, #At表示other
        # 其他原子或者不在列表中的原子，可以根据需要添加
    }
    # 获取输入元素编号对应的原子序数
    return heavyatom_atomic_numbers.get(element_number, "Unknown")

#upper
def Connect_U(matrix):
    mol = openbabel.OBMol()
    for i in range(matrix.shape[0]):
        for j in range(i):
            if matrix[i][j][2]!=0 and matrix[i][j][0]!=0 and matrix[i][j][1]!=0:

                print(matrix[i][j],i,j)

#lower
def Connect_L(matrix):
    k=0
    for i in range(matrix.shape[0]):
        for j in range(i):
            if matrix[j][i][2]!=0 and matrix[j][i][0]!=0 and matrix[j][i][1]!=0:
                k=k+1
                print(matrix[i][j],k)

#complex=upper+lower
def Connect_C(matrix):
    info=[]
#    k=0
    for i in range(matrix.shape[0]):
        for j in range(i):
            if matrix[i][j][2]!=0 and matrix[i][j][0]!=0 and matrix[i][j][1]!=0:
                if abs(matrix[i][j][0]-matrix[j][i][1])<0.3 and abs(matrix[i][j][1]-matrix[j][i][0])<0.3 and abs(matrix[i][j][2]-matrix[j][i][2])<0.3:
                    element_i = element_number_to_atomic_number(matrix[i][j][0])
                    element_j = element_number_to_atomic_number(matrix[i][j][1])
                    if element_i != 'Unknown' and element_j != 'Unknown':
                        info.append(((j, int(element_j)),(i, int(element_i)), int(matrix[i][j][2])))
                    else:
                        # Handle the 'Unknown' case here. Maybe log an error or continue.
                        pass
    return info

def Connect_Ndigo(matrix):
    info=[]
#    k=0
    for i in range(matrix.shape[0]):
        for j in range(i):
            if matrix[i][j][2]!=0 and matrix[i][j][0]!=0 and matrix[i][j][1]!=0:
                element_i = element_number_to_atomic_number(matrix[i][j][0])
                element_j = element_number_to_atomic_number(matrix[i][j][1])
                if element_i != 'Unknown' and element_j != 'Unknown':
                    info.append(((j, int(element_j)),(i, int(element_i)), int(matrix[i][j][2])))
                else:
                    # Handle the 'Unknown' case here. Maybe log an error or continue.
                    pass
    return info

def correct_bond_types(connections):
    from collections import defaultdict, Counter
    
    # 计算每个原子的连接次数
    atom_counts = Counter()
    for (atom1, element1), (atom2, element2), _ in connections:
        atom_counts[atom1] += 1
        atom_counts[atom2] += 1

    # 定义元素的典型最大键数
    max_bonds = {34: 2, 3: 1, 9: 1, 14: 4, 52: 2, 8: 2, 5: 3, 15: 3, 7: 3, 33: 3, 53: 1, 35: 1, 19: 1, 6: 4, 20: 2, 11: 1, 16: 1, 17: 2, 1: 1, 85: 1}

    # 确定需要修正的原子
    to_correct = set()
    for (atom_id, count) in atom_counts.items():
        element = None
        for conn in connections:
            if conn[0][0] == atom_id:
                element = conn[0][1]
                break
            elif conn[1][0] == atom_id:
                element = conn[1][1]
                break
        if element and count == max_bonds.get(element, 0):
            to_correct.add(atom_id)

    # 更新连接信息
    new_connections = []
    for (atom1, element1), (atom2, element2), bond_type in connections:
        if atom1 in to_correct or atom2 in to_correct:
            new_connections.append(((atom1, element1), (atom2, element2), 1))  # 强制为单键
        else:
            new_connections.append(((atom1, element1), (atom2, element2), bond_type))
    
    return new_connections

def create_mol_from_list_rdkit(connections, output_file='/path/to/output/mol_file.mol'):
    mol = Chem.RWMol()  # 使用可读写的分子对象

    atoms = {}

    # Step 1: 遍历连接信息，创建并设置原子的原子序数
    for connection in connections:
        for atom_info in connection[:2]:
            atom_id, atomic_num = atom_info
            if atom_id not in atoms:
                atom = Chem.Atom(int(atomic_num))
                atom_idx = mol.AddAtom(atom)  # 添加原子并获取索引
                atoms[atom_id] = atom_idx  # 保持原始的从1开始的索引
    print(atoms)
    # Step 2: 添加原子之间的键
    bond_type_dict = {1: Chem.BondType.SINGLE, 3: Chem.BondType.DOUBLE, 4: Chem.BondType.TRIPLE, 2: Chem.BondType.AROMATIC}
    for connection in connections:
        atom1_id, atom2_id = connection[0][0], connection[1][0]
        bond_type = bond_type_dict.get(connection[2], Chem.BondType.SINGLE)  # 默认为单键
        mol.AddBond(atoms[atom1_id] , atoms[atom2_id] , bond_type)  # 转换为从0开始的索引以适应RDKit
    for bond in mol.GetBonds():
        if not bond.IsInRing():  # 如果键不在环中
            if bond.GetBondType() == Chem.BondType.AROMATIC:
                # 假设芳香键应该是单键或双键，这里需要根据化学逻辑决定如何修正
                bond.SetBondType(Chem.BondType.SINGLE) 
    ssr = Chem.GetSymmSSSR(mol)
    aromatic_bonds = {4} 

    for ring in ssr:
        ring_list = list(ring)  # 将 _vecti 转换为 list
        if all(mol.GetBondBetweenAtoms(mol.GetAtomWithIdx(atom).GetIdx(), mol.GetAtomWithIdx(next_atom).GetIdx()).GetBondType() in aromatic_bonds for atom, next_atom in zip(ring_list, ring_list[1:] + [ring_list[0]])):
            for atom in ring:
                for bond in mol.GetAtomWithIdx(atom).GetBonds():
                    if bond.GetBondType() in aromatic_bonds:
                        bond.SetBondType(Chem.BondType.AROMATIC)
    # Step 3: 清理和标准化分子，确保所有的价数和结构都是正确的
    mol.UpdatePropertyCache(strict=False)
    mol = Chem.AddHs(mol)
    Chem.SanitizeMol(mol)

    # Step 4: 保存分子到.mol文件
    writer = Chem.SDWriter(output_file)
    writer.write(mol)
    writer.close()

def create_mol_from_list(connections, output_file=mol_file_path):
    mol = openbabel.OBMol()
    atoms = {}

    # Step 1: 遍历连接信息，创建并设置原子的原子序数
    for connection in connections:
        for atom_info in connection[:2]:
            atom_id, atomic_num = atom_info
            if atom_id not in atoms:
                atoms[atom_id] = atomic_num

    for atom_id, atomic_num in atoms.items():
        atom = mol.NewAtom()
        atom.SetAtomicNum(int(atomic_num))
        atoms[atom_id] = atom
    #print(atoms)

    # 按atom_id排序并创建原子
    # for atom_id in sorted(atoms.keys()):
    #     atomic_num = atoms[atom_id]
    #     atom = mol.NewAtom()
    #     atom.SetAtomicNum(int(atomic_num))
    #     atoms[atom_id] = atom
    # print(atoms)
    #Step 2: 添加原子之间的键

    for connection in connections:
        atom1_id, atom2_id = connection[0][0], connection[1][0]
        if connection[2]==1:
            bond_type = 1
        elif connection[2]==2:
            bond_type = 4
        elif connection[2]==3:
            bond_type = 2
        elif connection[2]==4:
            bond_type = 3
        mol.AddBond(atom1_id, atom2_id, bond_type)  # 添加键，+1因为OBMol中的索引是从1开始的

    # # # Step 3: (可选) 为分子生成3D坐标
    # builder = openbabel.OBBuilder()
    # builder.Build(mol)
    for bond in openbabel.OBMolBondIter(mol):
        if not bond.IsInRing():
            if bond.GetBondOrder() == 4:  # Open Babel 中芳香键可能被读为1.5
                bond.SetBondOrder(1)

    # Step 4: 保存分子到.mol文件
    obConversion = openbabel.OBConversion()
    obConversion.SetOutFormat("MOL")
    obConversion.WriteFile(mol, output_file)

def reset_mol_atom_info(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    output_lines = []
    zero_atom="   0  0  0  0  0  0  0  0  0  0  0  0\n"
    zero_bond="  0  0  0  0\n"
    for line in lines:
        # print(len(line))
        if line.strip().startswith('M END'):
            output_lines.append(line)
            break
        elif len(line)==70:
            newmol=line[:32]+zero_atom
            output_lines.append(newmol)
        elif len(line)==22:
            newmol=line[:10]+zero_bond
            output_lines.append(newmol)
        else :
            output_lines.append(line)
    with open(output_file, 'w') as f:
        f.writelines(output_lines)
        
def mol_generate(matrix,output_file):
    matrix=roundmatrix_cut(matrix)
    list_atom=Connect_C(matrix)
    list_atom=fix_atom_ids(list_atom)
    # list_atom=correct_bond_types(list_atom)
    create_mol_from_list(list_atom)
    #create_mol_from_list_rdkit(list_atom,output_file)
    input_mol=mol_file_path
    # input_mol2='/data/yakowei/research/validate/mol/mol_file/tmp1.mol'
    #output_mol='/data/yakowei/research/Final_Output/mol_file/64_1000_condi/0301/epoch_200/tmp.mol'
    openbabel_fixed(input_mol,output_file)
    reset_mol_atom_info(output_file,output_file)
    # rdkit_fixed(input_mol2, output_file)
    print('done')

def mol_generate_Ndigo(matrix,output_file):
    matrix=roundmatrix(matrix)
    list_atom=Connect_Ndigo(matrix)
    list_atom=fix_atom_ids(list_atom)
    # list_atom=correct_bond_types(list_atom)
    create_mol_from_list(list_atom)
    #create_mol_from_list_rdkit(list_atom,output_file)
    input_mol=mol_file_path
    # input_mol2='/data/yakowei/research/validate/mol/mol_file/tmp1.mol'
    #output_mol='/data/yakowei/research/Final_Output/mol_file/64_1000_condi/0301/epoch_200/tmp.mol'
    # openbabel_fixed(input_mol,output_file)
    reset_mol_atom_info(input_mol,output_file)
    # rdkit_fixed(input_mol2, output_file)
    print('done')

def rdkit_fixed(input_file, output_file):
    # 读取或创建分子结构
    mol = Chem.MolFromMolFile(input_file, sanitize=False)  # 先不要进行标准化处理

    if mol is None:
        print(f"Failed to read molecule from {input_file}")
        return

    try:
        # 强制清洗和更新分子的属性，确保隐式价数被计算
        Chem.SanitizeMol(mol)
        mol.UpdatePropertyCache(strict=False)

        # 添加明确的氢原子（Hs）
        mol = Chem.AddHs(mol)

        # 生成3D坐标
        AllChem.EmbedMolecule(mol)

        # 使用MMFF94力场优化分子结构
        AllChem.MMFFOptimizeMolecule(mol)

        # 在保存之前去除添加的氢原子
        mol_no_h = Chem.RemoveHs(mol)

        # 储存优化后的结构
        Chem.MolToMolFile(mol_no_h, output_file)

        print(f"Molecule optimized and saved to {output_file}")
    except Exception as e:
        print(f"Error during processing: {e}")

def openbabel_fixed(input_file, output_file):
    # 初始化Open Babel的转换器
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("mol", "mol")

    # 创建一个分子对象
    mol = openbabel.OBMol()

    # 读取输入的MOL文件
    if not obConversion.ReadFile(mol, input_file):
        print(f"Failed to read molecule from {input_file}")
        return

    # 添加氢原子
    mol.AddHydrogens()

    # 生成3D坐标
    builder = openbabel.OBBuilder()
    builder.Build(mol)

    # 进行能量最小化
    ff = openbabel.OBForceField.FindForceField('MMFF94')
    if ff.Setup(mol):
        ff.ConjugateGradients(250, 1.0e-9)
        ff.GetCoordinates(mol)

    # 删除所有氢原子
    mol.DeleteHydrogens()

    # 保存处理后的分子到输出文件
    if not obConversion.WriteFile(mol, output_file):
        print(f"Failed to write molecule to {output_file}")
        return

#rdkit virsion 2022.9.5
# def rdkit_draw(input_mol,output_img,sanitize=False):
#     mol = Chem.MolFromMolFile(input_mol, sanitize=sanitize)
#     if mol is not None:
#         # 直接计算2D坐标，跳过开克勒化步骤
#         Chem.rdDepictor.Compute2DCoords(mol)
#         # 绘制分子的2D图形
#         fig = plt.figure(figsize=(10, 10), dpi=300)

#         # 绘制分子，确保足够的边距以避免裁剪
#         ax = fig.add_subplot(111)
#         ax.axis('off')  # 不显示轴
#         Draw.MolToMPL(mol, ax=ax, bbox_inches='tight')

#         # 保存图像，同时指定bbox_inches='tight'来减少空白边缘
#         plt.savefig(output_img, bbox_inches='tight')
#         plt.clf()
#         plt.close(fig)

#     else:
#         print("无法从文件加载分子。请检查文件路径和格式。")

#rdkit virsion 2024.9.5
def rdkit_draw(input_mol, output_img, sanitize=False):
    # Load molecule from file
    mol = Chem.MolFromMolFile(input_mol, sanitize=sanitize)
    if mol is not None:
        # Compute 2D coordinates for visualization
        Chem.rdDepictor.Compute2DCoords(mol)

        # Draw the molecule to an SVG file
        svg = Draw.MolToImage(mol, size=(1000, 1000))
        
        # Save the image
        svg.save(output_img)
    else:
        print("Unable to load molecule from file. Please check the file path and format.")

def rdkit_draw_ugly(input_mol,output_img,sanitize=False):
    mol = Chem.MolFromMolFile(input_mol, sanitize=sanitize)
    if mol is not None:
        Chem.rdDepictor.Compute2DCoords(mol)
        img = Draw.MolToImage(mol, size=(600, 600))
        img.save(output_img) 
    else:
        print("无法从文件加载分子。请检查文件路径和格式。")

def fix_atom_ids(connections):
    # 找到所有独特的原子ID
    unique_ids = sorted(set(atom_id for conn in connections for atom_id in (conn[0][0], conn[1][0])))

    # 创建一个新的ID映射
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids, start=1)}

    # 使用新的ID映射创建更新的连接列表
    fixed_connections = [
        ((id_mapping[atom1_info[0]], atom1_info[1]), 
         (id_mapping[atom2_info[0]], atom2_info[1]), 
         bond_order)
        for (atom1_info, atom2_info, bond_order) in connections
    ]

    return fixed_connections

if __name__=='__main__' :
    import faulthandler
    faulthandler.enable()

    # matrix=np.load('/data/yakowei/research/Final_Output/mol_file/64_1000_condi/1000step/np_file/img_999.npy')
    # matrix=roundmatrix_cut(matrix)
    # for i in range(40):
    #     for j in range(40):
    #         if matrix[i][j][2]>0.5:
    #             print(i,j,matrix[i][j])
    # a=Connect_C(matrix)
    # print(a)
    # a=fix_atom_ids(a)
    # print(a)
    # create_mol_from_list(a, output_file='/data/yakowei/research/validate/mol/mol_file/tmp.mol')
    # mol_generate_Ndigo(matrix,output_file='/data/yakowei/research/test/test_0425/step_999.mol')
    # input_mol='./mol_file/img_1.mol'
    output_mol='/data/yakowei/research/Final_Output/mol_file/64_1000_condi/1000step/mol_file/img_935_fix.mol'
    output_img='step_935.png'
    # openbabel_fixed(input_mol,output_mol)
    rdkit_draw_ugly(output_mol,output_img,sanitize=False)

    # input_file="/data/yakowei/research/Final_Output/mol_file/64_1000_condi/condi/epoch_270/data_condi/condi_9/img_37.mol"
    # output_file="/data/yakowei/research/Final_Output/mol_file/64_1000_condi/condi/test/rdkit_fix/img_0.mol"
    # rdkit_fixed(input_file, output_file)