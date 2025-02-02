import rdkit 
import torch 
import multiprocessing 
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_all_atom(smi_list) :
    atom_list = []
    for smi in smi_list :
        mol = rdkit.Chem.MolFromSmiles(smi) 
        for atom in mol.GetAtoms() :
            symbol = atom.GetSymbol() 
            if symbol not in atom_list :
                atom_list.append(symbol)
    return atom_list

def get_smi_list(path) :
    with open(path, 'r') as file :
        return [smi[:-1] for smi in file.readlines()]

def get_atom_pos(smi) :
    mol = rdkit.Chem.MolFromSmiles(smi)
    mol_h = rdkit.Chem.AddHs(mol)
    rdkit.Chem.rdDistGeom.EmbedMolecule(mol_h)
    conformer = mol_h.GetConformer()
    atom_pos = conformer.GetPositions()
    return atom_pos[:count_atom(smi)]

def parallel_f(f, input_list) :
    pool = multiprocessing.Pool()
    return pool.map(f, input_list)


def count_atom(smi) :
    return rdkit.Chem.MolFromSmiles(smi).GetNumAtoms()

def normalize(coor) :
    x, y, z = coor[0]
    return coor - [x,y,z]

def pad(coor, longest_coor) :
    zeros = np.zeros((longest_coor - coor.shape[0], 3))
    return np.concatenate((coor, zeros), axis=0)

# def encode_smi(smi) :
    
def replace_atom(smi) :
    return smi.replace('Cl', 'X').replace('Br', 'Y').replace('Na', 'Z').replace('Ba', 'T')


def get_dic(smi_list) :
    smi_dic = {'<P>': 0, '<E>' : 1}
    i = 2
    for smi in smi_list :
        for char in smi :
            if char not in smi_dic :
                smi_dic[char] = i
                i += 1
    return smi_dic


def encode_smi(smi, smi_dic) :
    smi = [smi_dic[c] for c in smi] 
    smi = smi + [smi_dic['<E>']]
    return smi 

def pad_smi(smi, longest_smi, smi_dic) :
    return smi + [smi_dic['<P>']] * (longest_smi - len(smi))



class MyDataset(torch.utils.data.Dataset) : 
    def __init__(self, smint_list, coor_list) : 
        self.smint_list = smint_list
        self.coor_list = coor_list

    def __len__(self) :
        return len(self.smint_list)
    
    def __getitem__(self, idx) :
        return torch.tensor(self.smint_list[idx],dtype = torch.long), torch.tensor(self.coor_list[idx], dtype = torch.float)  
    


