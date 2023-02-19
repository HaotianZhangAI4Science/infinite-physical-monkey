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