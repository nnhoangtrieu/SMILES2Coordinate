import rdkit 
import torch 
import multiprocessing 

def get_dic(smi_list) :
    smi_dic = {'<END>': -1, '<PAD>':0}
    count = 1
    for smi in smi_list :
        mol = rdkit.Chem.MolFromSmiles(smi) 
        for atom in mol.GetAtoms() :
            symbol = atom.GetSymbol() 
            if symbol not in smi_dic :
                smi_dic[symbol] = count 
                count += 1 
    return smi_dic

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
    return torch.tensor(coor - [x,y,z])

def pad(coor, longest_coor) :
    zeros = torch.zeros(longest_coor - coor.size(0), 3)
    return torch.cat((coor, zeros), dim = 0)


# def encode_smi(smi) :
    



class MyDataset(torch.utils.data.Dataset) : 
    def __init__(self, smi_list, coor_list) : 
        self.smi_list = smi_list
        self.coor_list = coor_list

    def __len__(self) :
        return len(self.smi_list)
    
    def __getitem__(self, idx) :
        return self.smi_list[idx], self.coor_list[idx]