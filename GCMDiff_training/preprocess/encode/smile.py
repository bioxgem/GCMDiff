from rdkit import Chem
import numpy as np
# from rdkit.Chem import Draw

def encoding(labels):
    unique_labels = list(labels)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels,start=1)}
    encoded_labels = [label_mapping[label] for label in labels]
    return encoded_labels, label_mapping
def get_label_for_string(input_string, label_mapping):
    for label, index in label_mapping.items():
        if input_string.lower() == label.lower():
            return index
    return 0

heavyatom=['Se', 'Li', 'F', 'Si', 'Te', 'O', 'B', 'P', 'N', 'As', 'I', 'Br', 'K', 'C', 'Ca', 'Na', 'S', 'Cl', 'H','other']
#["C", "N", "O", "F", "S", "Cl", "Br", "P", "I", "B", "Na", "Si", "Se", "H", "K", "Li", "Te", "Re", "Zn", "As", "*", "Au", "Pt", "Fe", "Sn", "Hg", "Ag", "V", "Ru", "Ni", "Sb", "Ca", "Cu", "Gd", "Pd", "Sr", "W", "Nb", "Mn", "Mo", "Tc", "Ar", "Os", "Co", "Bi", "Al"]
encoded_labels_example, heavyatom_mapping = encoding(heavyatom)

#smile="COc1ccc(COc2ccc(Cn3c(N)nc4cc(cnc34)-c3cnn(C)c3)cc2OC)cn1"

def get_smile_atoms(smile):
    #(smile編號,encode編號)
    mol=Chem.MolFromSmiles(smile)
    infoMatrix=np.zeros((mol.GetNumAtoms(),2))
    i=0
    for atom in mol.GetAtoms():
        infoMatrix[i]=[atom.GetIdx(),get_label_for_string(atom.GetSymbol(),heavyatom_mapping)]
        i=i+1
    return infoMatrix
def encode_bond_type(bond_type):
    if bond_type==1.5:
        return 2
    elif bond_type==2:
        return 3
    elif bond_type==3:
        return 4
    elif bond_type==1:
        return 1
    else :
        return 0
def get_smile_bounds(smile):
    #(begin smile編號,end smile編號,begin encode編號,end encode編號,bond type)
    mol=Chem.MolFromSmiles(smile)
    infoMatrix=np.zeros((mol.GetNumBonds(),5))
    i=0
    for bond in mol.GetBonds():
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()
        atom1 = mol.GetAtomWithIdx(atom1_idx)
        atom2 = mol.GetAtomWithIdx(atom2_idx)
        atom1=get_label_for_string(atom1.GetSymbol(),heavyatom_mapping)
        atom2=get_label_for_string(atom2.GetSymbol(),heavyatom_mapping)
        bond_type = bond.GetBondTypeAsDouble()
        bond_type=encode_bond_type(bond_type)
        infoMatrix[i]=[atom1_idx,atom2_idx,atom1,atom2,bond_type]
        i=i+1
    return infoMatrix

def check_and_get_bondtype(matrix, idx1, idx2):
    for row in matrix:
        if row[0] == idx1 and row[1] == idx2:
            return row[4]  # 返回該行的第五個值
        elif row[0] == idx2 and row[1] == idx1:
            return row[4]
    return 0

def smile2graph(smile):
    #(begin encode編號,end encode編號,bond type)
    mol=Chem.MolFromSmiles(smile)
    num_atoms = mol.GetNumAtoms()
    graph=np.zeros((40,40,3))
    # graph=np.zeros((num_atoms,num_atoms,3))
    atom=get_smile_atoms(smile)
    bond=get_smile_bounds(smile) 
    for i in range(mol.GetNumAtoms()):
        for j in range(mol.GetNumAtoms()):
                graph[i][j]=[atom[i][1]/20,atom[j][1]/20,check_and_get_bondtype(bond,i,j)/4]
                #graph[i][j]=[atom[i][1],atom[j][1],check_and_get_bondtype(bond,i,j)]
    return graph


# a=smile2graph(smile)
# print(a)
