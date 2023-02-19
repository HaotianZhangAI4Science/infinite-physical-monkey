from utils import *
#get_pocket, relax_inpocket
from glob import glob
from tqdm import tqdm
import os.path as osp
from rdkit.Chem.rdMolAlign import CalcRMS, GetBestRMS
from rdkit.Chem import AllChem

def get_best_rmsd(gen_mol, ref_mol):
    gen_mol = Chem.RemoveHs(gen_mol)
    ref_mol = Chem.RemoveHs(ref_mol)
    rmsd = GetBestRMS(gen_mol, ref_mol)
    return rmsd

def rmradical(mol):
    for atom in mol.GetAtoms():
        atom.SetNumRadicalElectrons(0)
    return mol
    
def get_rmsd(gen_mol, ref_mol):
    gen_mol = Chem.RemoveHs(gen_mol)
    ref_mol = Chem.RemoveHs(ref_mol)
    rmsd = CalcRMS(gen_mol, ref_mol)
    return rmsd

# def docked_rmsd(ref_mol, docked_mols):
#     rmsd_list  =[]
#     for mol in docked_mols:
#         clean_mol = rmradical(mol)
#         rightref = AllChem.AssignBondOrdersFromTemplate(clean_mol, ref_mol) #(template, mol)
#         rmsd = CalcRMS(rightref,clean_mol)
#         rmsd_list.append(rmsd)
#     return rmsd_list

def docked_rmsd(ref_mol, docked_mol):
    ref_mol = Chem.RemoveHs(ref_mol)
    docked_mol = Chem.RemoveHs(docked_mol)
    clean_mol = rmradical(docked_mol)
    rightref = AllChem.AssignBondOrdersFromTemplate(ref_mol, docked_mol) #(template, mol)
    rmsd = CalcRMS(rightref,clean_mol)
    return rmsd

from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

def rdkit_conf(tgt_mol, num_confs=1000, seed=42, MMFF=False):
    mol = copy.deepcopy(tgt_mol)
    mol = Chem.AddHs(mol)
    allconformers = AllChem.EmbedMultipleConfs(
        mol, numConfs=num_confs, randomSeed=seed, clearConfs=True
    )
    sz = len(allconformers)
    if MMFF:
        for i in range(sz):
            try:
                AllChem.MMFFOptimizeMolecule(mol, confId=i)
            except:
                print(f'failed to Optimize {i} conformer')
                continue
    mol = Chem.RemoveHs(mol)
    return mol

def confs2mols(mol):
    mols = []
    for conf in mol.GetConformers():
        mol2d = copy.deepcopy(mol)
        mol2d.RemoveAllConformers()
        mol2d.AddConformer(conf)
        mols.append(mol2d)
    return mols


### perform UFF and get the mols
core_base = './coreset'
targets = glob(osp.join(core_base,'*'))
for target in targets:
    target_name = target.split('/')[-1]
    protein = osp.join(target,f'{target_name}_protein.pdb')
    ori_mol = read_sdf(osp.join(target,f'{target_name}_ligand.sdf'))[0]
    rdkit_mols = read_sdf(osp.join(target,f'{target_name}_rdkit.sdf'))
    ori_coords = ori_mol.GetConformer().GetPositions().astype(np.float32)
    rec_mol = Chem.MolFromPDBFile(protein)
    pocket_uff_mols = []
    molpairs = []
    rec_mol = get_pocket(ori_mol,rec_mol,max_dist=10)
    for rd_idx, rd_mol in enumerate(rdkit_mols):
        _coords = rd_mol.GetConformer().GetPositions().astype(np.float32)
        _coords = _coords - _coords.mean(axis=0) 
        M = rand_rotation_matrix()
        _coords = np.dot(_coords,M)
        _coords = _coords + ori_coords.mean(axis=0)
        rand_mol = set_rdmol_positions(rd_mol,_coords)
        uff_mol = relax_inpocket_UFF(rand_mol,rec_mol)
        pocket_uff_mols.append(uff_mol)
        rmsd = CalcRMS(ori_mol,uff_mol)
        molpairs.append((rmsd,rand_mol))
    write_pkl(molpairs,osp.join(target,'scoring_random_result.pkl'))
    write_sdf(pocket_uff_mols,osp.join(target,'uff_optmols.sdf'))