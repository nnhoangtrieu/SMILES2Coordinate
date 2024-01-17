import rdkit 
import torch 

def read_smi_txt(path) :
    with open(path, 'r') as file : 
        smi_list = [smi for smi in file.readlines()]
    return smi_list



def get_coor(path) :
    coor_list = []
    supplier = rdkit.Chem.SDMolSupplier(path)
    
    
    for mol in supplier:
        print(mol)
        coor = []
        if mol is not None:
            conformer = mol.GetConformer()
            for atom in mol.GetAtoms():
                atom_idx = atom.GetIdx()
                x, y, z = conformer.GetAtomPosition(atom_idx)
                coor_atom = list((x,y,z))
                coor.append(coor_atom)
        coor_list.append(coor)

    # Replace invalid idx
    for i, coor in enumerate(coor_list):
        if len(coor) == 0 :
            if i == 0 :
                coor_list = coor_list[1:]
            coor_list[i] = coor_list[i-1]


    return coor_list

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

def count_atom(smi) :
    return rdkit.Chem.MolFromSmiles(smi).GetNumAtoms()

def normalize(coor) :
    x, y, z = coor[0]
    return torch.tensor(coor - [x,y,z])


def encode_smi(smi) :
    
