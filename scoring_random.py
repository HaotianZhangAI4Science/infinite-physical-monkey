from utils import *
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

from docking_utils import *

### scoring and assign the scoring function
core_base = '/home/haotian/molecules_confs/small_molecules_conformation/infinite_physical_monkey/coreset'
result_base = '/home/haotian/molecules_confs/small_molecules_conformation/infinite_physical_monkey/result'
import numpy as np
targets = np.sort(glob(osp.join(core_base,'*')))
all_rmsd = []
rand_rotation = 1
for target in tqdm(targets):
    try:
        target_name = target.split('/')[-1]
        print(target_name)
        protein_prepare1 = osp.join(target,f'{target_name}_protein.pdbqt')
        result_target = osp.join(result_base, target_name)
        if osp.exists(osp.join(result_target,'scoring_random_result.pkl')):
            continue
        protein_prepare2 = osp.join(result_target,f'{target_name}_protein.pdbqt')
        shutil.copyfile(protein_prepare1, protein_prepare2)
        ori_mol = read_sdf(osp.join(target,f'{target_name}_ligand.sdf'))[0]
        rdkit_mols = read_sdf(osp.join(target,f'{target_name}_rdkit.sdf'))
        ori_coords = ori_mol.GetConformer().GetPositions().astype(np.float32)
        work_dir = osp.join(result_target,'pocket_random')
        os.makedirs(work_dir,exist_ok=True)
        if not osp.exists(osp.join(work_dir,f'{target_name}_protein.pdbqt')):
            shutil.copy(protein_prepare2,work_dir)

        pocket_rand_mols = []
        molpairs = []
        rand_mols = []
        for rd_idx, rd_mol in enumerate(rdkit_mols):
            try:
                _coords = rd_mol.GetConformer().GetPositions().astype(np.float32)
                _coords = _coords - _coords.mean(axis=0) 
                M = rand_rotation_matrix()
                _coords = np.dot(_coords,M)
                _coords = _coords + ori_coords.mean(axis=0)
                rand_mol = set_rdmol_positions(rd_mol,_coords)
                pocket_rand_mols.append(rand_mol)
                write_sdf([rand_mol],osp.join(work_dir,f'{rd_idx}.sdf'))
                prepare_ligand(work_dir,f'{rd_idx}.sdf')
                centroid = ori_coords.mean(axis=0)
                fake_centroid = _coords.mean(axis=0)
                score = scoring_with_sdf(work_dir, protein_prepare2.split('/')[-1], f'{rd_idx}.pdbqt', centroid)
                rmsd = CalcRMS(ori_mol,rand_mol)
                molpairs.append((score,rmsd,rand_mol))
                rand_mols.append(rand_mol)
            except:
                ...
        write_pkl(molpairs,osp.join(result_target,'scoring_random_result.pkl'))
        write_sdf(rand_mols,osp.join(result_target,'pocket_random.sdf'))
    except:
        print('failed')