import scipy as sp
import numpy as np
import pickle
from rdkit import Chem
import copy
from rdkit.Chem import rdMolAlign as MA
from rdkit.Chem import AllChem

def read_pkl(pkl_file):
    with open(pkl_file,'rb') as f:
        data = pickle.load(f)
    return data

def write_pkl(data_list, pkl_file):
    with open(pkl_file, 'wb') as f:
        pickle.dump(data_list, f)

def write_sdf(mol_list,file):
    writer = Chem.SDWriter(file)
    for i in mol_list:
        writer.write(i)
    writer.close()

def read_sdf(file):
    supp = Chem.SDMolSupplier(file)
    return [i for i in supp]

def set_rdmol_positions(rdkit_mol, pos):
    rdkit_mol = Chem.RemoveHs(rdkit_mol)
    assert rdkit_mol.GetConformer(0).GetPositions().shape[0] == pos.shape[0]
    mol = copy.deepcopy(rdkit_mol)
    for i in range(pos.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, pos[i].tolist())
    return mol

from rdkit.Chem.rdMolAlign import CalcRMS, GetBestRMS
def get_best_rmsd(gen_mol, ref_mol):
    gen_mol = Chem.RemoveHs(gen_mol)
    ref_mol = Chem.RemoveHs(ref_mol)
    rmsd = GetBestRMS(gen_mol, ref_mol)
    return rmsd

def get_rmsd(gen_mol, ref_mol):
    gen_mol = Chem.RemoveHs(gen_mol)
    ref_mol = Chem.RemoveHs(ref_mol)
    rmsd = CalcRMS(gen_mol, ref_mol)
    return rmsd

def rmradical(mol):
    for atom in mol.GetAtoms():
        atom.SetNumRadicalElectrons(0)
    return mol

def docked_rmsd(ref_mol, docked_mols):
    rmsd_list  =[]
    for mol in docked_mols:
        clean_mol = rmradical(mol)
        rightref = AllChem.AssignBondOrdersFromTemplate(clean_mol, ref_mol) #(template, mol)
        rmsd = CalcRMS(rightref,clean_mol)
        rmsd_list.append(rmsd)
    return rmsd_list

def get_rmsd_mat(gen_mols, ref_mols):
    rmsd_mat = np.zeros([len(ref_mols), len(gen_mols)], dtype=np.float32)
    for i, gen_mol in enumerate(gen_mols):
        gen_mol_c = copy.deepcopy(gen_mol)
        for j, ref_mol in enumerate(ref_mols):
            ref_mol_c = copy.deepcopy(ref_mol)
            rmsd_mat[j, i] = get_best_rmsd(gen_mol_c, ref_mol_c)
    return rmsd_mat

def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.
    
    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    
    if randnums is None:
        randnums = np.random.uniform(size=(3,))
        
    theta, phi, z = randnums
    
    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.
    
    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.
    
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )
    
    st = np.sin(theta)
    ct = np.cos(theta)
    
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


from sklearn.cluster import KMeans
def clustering(mols, N=100):
    total_sz = len(mols)
    rdkit_coords_list = []
    for mol in mols:
        coords = mol.GetConformer().GetPositions().astype(np.float32)
        rdkit_coords_list.append(coords)
    # clustering
    rdkit_coords_flatten = np.array(rdkit_coords_list).reshape(total_sz, -1)
    cluster_size = N
    kmeans = KMeans(n_clusters=cluster_size, random_state=42).fit(rdkit_coords_flatten)
    ids = kmeans.predict(rdkit_coords_flatten)
    # get cluster center
    center_coords = kmeans.cluster_centers_
    coords_list = [center_coords[i].reshape(-1,3) for i in range(cluster_size)]
    print(f'len(coords_list): {len(coords_list)}, \n coords: {coords_list[0]}')
    return coords_list

from scipy.spatial.transform import Rotation
def align_clustering(mols, N=100):
    total_sz = len(mols)
    rdkit_coords_list = []
    # normalize for the first molecules
    tgt_coords = mols[0].GetConformers()[0].GetPositions().astype(np.float32)
    tgt_coords = tgt_coords - np.mean(tgt_coords, axis=0)

    for mol in mols:
        _coords = mol.GetConformer().GetPositions().astype(np.float32)
        _coords = _coords - _coords.mean(axis=0)
        _R, _score = Rotation.align_vectors(_coords, tgt_coords)
        rdkit_coords_list.append(np.dot(_coords, _R.as_matrix()))
    # clustering
    rdkit_coords_flatten = np.array(rdkit_coords_list).reshape(total_sz, -1)
    cluster_size = N
    kmeans = KMeans(n_clusters=cluster_size, random_state=42).fit(rdkit_coords_flatten)
    ids = kmeans.predict(rdkit_coords_flatten)
    # get cluster center
    center_coords = kmeans.cluster_centers_
    coords_list = [center_coords[i].reshape(-1,3) for i in range(cluster_size)]
    print(f'len(coords_list): {len(coords_list)}')
    return coords_list

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


from rdkit import Chem 
from rdkit.Chem import AllChem
def get_rd_atom_res_id(rd_atom):
    '''
    Return an object that uniquely
    identifies the residue that the
    atom belongs to in a given PDB.
    '''
    res_info = rd_atom.GetPDBResidueInfo()
    return (
        res_info.GetChainId(),
        res_info.GetResidueNumber()
    )
def get_pocket(lig_mol, rec_mol, max_dist=8):
    lig_coords = lig_mol.GetConformer().GetPositions()
    rec_coords = rec_mol.GetConformer().GetPositions()
    dist = sp.spatial.distance.cdist(lig_coords, rec_coords)

    # indexes of atoms in rec_mol that are
    #   within max_dist of an atom in lig_mol
    pocket_atom_idxs = set(np.nonzero((dist < max_dist))[1])

    # determine pocket residues
    pocket_res_ids = set()
    for i in pocket_atom_idxs:
        atom = rec_mol.GetAtomWithIdx(int(i))
        res_id = get_rd_atom_res_id(atom)
        pocket_res_ids.add(res_id)

    # copy mol and delete atoms
    pkt_mol = rec_mol
    pkt_mol = Chem.RWMol(pkt_mol)
    for atom in list(pkt_mol.GetAtoms()):
        res_id = get_rd_atom_res_id(atom)
        if res_id not in pocket_res_ids:
            pkt_mol.RemoveAtom(atom.GetIdx())

    Chem.SanitizeMol(pkt_mol)
    return pkt_mol

def relax_inpocket_UFF(rd_mol, rec_mol):
    rd_mol = Chem.RWMol(rd_mol)
    uff_mol = Chem.CombineMols(rec_mol, rd_mol)
    try:
        Chem.SanitizeMol(uff_mol)
    except Chem.AtomValenceException:
        print('Invalid valence')
    except (Chem.AtomKekulizeException, Chem.KekulizeException):
        print('Failed to kekulize')
    try:
        uff = AllChem.UFFGetMoleculeForceField(
                        uff_mol, confId=0, ignoreInterfragInteractions=False
                    )
        uff.Initialize()
        for i in range(rec_mol.GetNumAtoms()): # Fix the rec atoms
            uff.AddFixedPoint(i)
        converged = False
        n_iters=200
        n_tries=2
        while n_tries > 0 and not converged:
            print('.', end='', flush=True)
            converged = not uff.Minimize(maxIts=n_iters)
            n_tries -= 1
        print(flush=True)
        print("Performed UFF with binding site...")
    except:
        print('Skip UFF...')
    coords = uff_mol.GetConformer().GetPositions()
    rd_conf = rd_mol.GetConformer()
    for i, xyz in enumerate(coords[-rd_mol.GetNumAtoms():]):
        rd_conf.SetAtomPosition(i, xyz)
    tmp_mol = copy.deepcopy(rd_mol)
    tmp_mol.RemoveAllConformers()
    tmp_mol.AddConformer(rd_conf)
    return tmp_mol

def relax_inpocket_MMFF(rd_mol, rec_mol):
    rd_mol = Chem.RWMol(rd_mol)
    mmff_mol = Chem.CombineMols(rec_mol, rd_mol)
    try:
        Chem.SanitizeMol(mmff_mol)
    except Chem.AtomValenceException:
        print('Invalid valence')
    except (Chem.AtomKekulizeException, Chem.KekulizeException):
        print('Failed to kekulize')
    try:
        pyMP = AllChem.MMFFGetMoleculeProperties(mmff_mol)
        mmff = AllChem.MMFFGetMoleculeForceField(
                        mmff_mol, pyMP, confId=0, ignoreInterfragInteractions=False
                    )
        mmff.Initialize()
        for i in range(rec_mol.GetNumAtoms()): # Fix the rec atoms
            mmff.AddFixedPoint(i)
        converged = False
        n_iters=200
        n_tries=2
        while n_tries > 0 and not converged:
            print('.', end='', flush=True)
            converged = not mmff.Minimize(maxIts=n_iters)
            n_tries -= 1
        print(flush=True)
        print("Performed MMFF with binding site...")
    except:
        print('Skip MMFF...')
    coords = mmff_mol.GetConformer().GetPositions()
    rd_conf = rd_mol.GetConformer()
    for i, xyz in enumerate(coords[-rd_mol.GetNumAtoms():]):
        rd_conf.SetAtomPosition(i, xyz)
    tmp_mol = copy.deepcopy(rd_mol)
    tmp_mol.RemoveAllConformers()
    tmp_mol.AddConformer(rd_conf)
    return tmp_mol


def get_torsions(m):
    m = Chem.RemoveHs(m)
    torsionList = []
    torsionSmarts = "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"
    torsionQuery = Chem.MolFromSmarts(torsionSmarts)
    matches = m.GetSubstructMatches(torsionQuery)
    for match in matches:
        idx2 = match[0]
        idx3 = match[1]
        bond = m.GetBondBetweenAtoms(idx2, idx3)
        jAtom = m.GetAtomWithIdx(idx2)
        kAtom = m.GetAtomWithIdx(idx3)
        for b1 in jAtom.GetBonds():
            if b1.GetIdx() == bond.GetIdx():
                continue
            idx1 = b1.GetOtherAtomIdx(idx2)
            for b2 in kAtom.GetBonds():
                if (b2.GetIdx() == bond.GetIdx()) or (b2.GetIdx() == b1.GetIdx()):
                    continue
                idx4 = b2.GetOtherAtomIdx(idx3)
                # skip 3-membered rings
                if idx4 == idx1:
                    continue
                # skip torsions that include hydrogens
                if (m.GetAtomWithIdx(idx1).GetAtomicNum() == 1) or (
                    m.GetAtomWithIdx(idx4).GetAtomicNum() == 1
                ):
                    continue
                if m.GetAtomWithIdx(idx4).IsInRing():
                    torsionList.append((idx4, idx3, idx2, idx1))
                    break
                else:
                    torsionList.append((idx1, idx2, idx3, idx4))
                    break
            break
    return torsionList

from rdkit.Chem import rdMolTransforms
def SetDihedral(conf, atom_idx, new_vale):
    rdMolTransforms.SetDihedralRad(
        conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale
    )

def single_conf_gen_bonds(tgt_mol, num_confs=1000, seed=42):
    mol = copy.deepcopy(tgt_mol)
    rotable_bonds = get_torsions(mol)
    bond_mols = []
    for i in range(num_confs):
        tmp_mol = copy.deepcopy(mol)
        values = 3.1415926 * 2 * np.random.rand(len(rotable_bonds))
        for idx in range(len(rotable_bonds)):
            SetDihedral(tmp_mol.GetConformers()[0], rotable_bonds[idx], values[idx])
        bond_mols.append(tmp_mol)

    return bond_mols

def optimize_mol(mol):
    mol = Chem.AddHs(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    return Chem.RemoveHs(mol)

import torch
def reconstruct_xyz(d_target, edge_index, init_pos, edge_order=None, alpha=0.5, mu=0, step_size=None, num_steps=None, verbose=0):
    assert torch.is_grad_enabled, 'the optimization procedure needs gradient to iterate'
    step_size = 8.0 if step_size is None else step_size
    num_steps = 1000 if num_steps is None else num_steps
    pos_vecs = []

    d_target = d_target.view(-1)
    pos = init_pos.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([pos], lr=step_size)

    #different hop contributes to different loss 
    if edge_order is not None:
        coef = alpha ** (edge_order.view(-1).float() - 1)
    else:
        coef = 1.0
    
    if mu>0:
        noise = torch.randn_like(coef) * coef * mu + coef
        noise = torch.clamp_min(coef+noise, min=0)
    
    for i in range(num_steps):
        optimizer.zero_grad()
        d_new = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
        loss = (coef * (d_target - d_new)**2).sum()
        loss.backward()
        optimizer.step()
        pos_vecs.append(pos.detach().cpu())
        avg_loss = loss.item() / d_target.size(0)
        #if verbose & (i%10 == 0):
        #    print('Reconstruction Loss: AvgLoss %.6f' % avg_loss)
    pos_vecs = torch.stack(pos_vecs, dim=0)
    avg_loss = loss.item() / d_target.size(0)
    if verbose:
        print('Reconstruction Loss: AvgLoss %.6f' % avg_loss)
    
    return pos_vecs, avg_loss

class Reconstruct_xyz(object):
    
    def __init__(self, alpha=0.5, mu=0, step_size=8.0, num_steps=1000, verbose=0):
        super().__init__()
        self.alpha = alpha
        self.mu = mu
        self.step_size = step_size
        self.num_stpes = num_steps
        self.verbose = verbose
    
    def __call__(self, d_target, edge_index, init_pos, edge_order=None):
        return reconstruct_xyz(
            d_target, edge_index, init_pos, edge_order, 
            alpha=self.alpha, 
            mu = self.mu, 
            step_size = self.step_size,
            num_steps = self.num_stpes,
            verbose = self.verbose
        )

def cluster_smis(mols):
    smis2mols = {}
    for mol in mols:
        smi = Chem.MolToSmiles(mol)
        try:
            smis2mols[smi].append(mol)
        except:
            smis2mols[smi] = []
    return smis2mols