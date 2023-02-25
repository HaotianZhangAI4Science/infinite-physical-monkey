from utils import *
from tqdm import tqdm
from torch_geometric.data import Data, Batch
import os.path as osp

# dataset could be extraced from the guildline in README.md, or just open the statistic_drugs.pkl 
dataset = read_pkl('./train_data_39k.pkl')
pair_dict = {}
test_mols = []
for data_idx in tqdm(range(len(dataset))):
    try:
        mol = Chem.RemoveHs(dataset[data_idx].rdmol)
        Chem.SanitizeMol(mol)
        test_mols.append(mol)
        conf = mol.GetConformer().GetPositions()
        for bond in mol.GetBonds():
            bond_type = bond.GetBondType()
            bond_type = str(bond_type)
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            begin_type = mol.GetAtomWithIdx(begin_idx).GetSymbol()
            end_type = mol.GetAtomWithIdx(end_idx).GetSymbol()
            begin_pos = conf[begin_idx]
            end_pos = conf[end_idx]

            dis = np.around(np.linalg.norm(begin_pos - end_pos),8)
            try:
                pair_dict[begin_type + f'_{bond_type}_' + end_type].append(dis)
                pair_dict[end_type + f'_{bond_type}_' + begin_type].append(dis)
            except:
                pair_dict[begin_type + f'_{bond_type}_' + end_type] = []
                pair_dict[end_type + f'_{bond_type}_' + begin_type] = []
                pair_dict[begin_type + f'_{bond_type}_' + end_type].append(dis)
                pair_dict[end_type + f'_{bond_type}_' + begin_type].append(dis)
    except:
        ...

statis_mean = {}
statis_std = {}
bond_types = list(pair_dict.keys())
for bond_type in bond_types:
    statis_mean[bond_type] = np.mean(pair_dict[bond_type])
    statis_std[bond_type] = np.std(pair_dict[bond_type])

# generate infinite physical monkey molecules
test_mols = read_pkl('./test_data_200.pkl')
smis2mols = cluster_smis(test_mols)
small_molecules_base = './GEOM_drugs'
recon3d = Reconstruct_xyz()
smis2rands = {}
smi_key = list(smis2mols.keys())
for smi_id in tqdm(range(len(smi_key))):
    smi = smi_key[smi_id]
    mol = smis2mols[smi][0]
    num_nodes = mol.GetNumAtoms()
    smis2rands[smi] = []
    src_list = []    
    dst_list = []
    disttype_list = []
    for bond in mol.GetBonds():
        u_idx = bond.GetBeginAtomIdx()
        v_idx = bond.GetEndAtomIdx()
        u = bond.GetBeginAtom()
        v = bond.GetEndAtom()
        u_type = u.GetSymbol()
        v_type = v.GetSymbol()
        bond_type = bond.GetBondType()
        bond_type = str(bond_type)
        dist_type = u_type + f'_{bond_type}_' + v_type
        src_list.extend([u_idx,v_idx])
        dst_list.extend([v_idx,u_idx])
        disttype_list.append(dist_type)
        disttype_list.append(dist_type)
    init_pos = torch.tensor(mol.GetConformer().GetPositions())    
    edge_index = torch.tensor([src_list, dst_list])

    confs_all = []
    data_list = []
    for i in tqdm(range(2000)):
        dist_list = []
        for dist_type in disttype_list:
            distance = np.random.normal(statis_mean[dist_type],statis_std[dist_type])
            dist_list.append(distance)
        d_target = torch.tensor(dist_list)
        data_list.append(Data(d_target=d_target,edge_index=edge_index, num_nodes=num_nodes, init_pos=init_pos))
    
    for batch_id in range(2000//500):
        batch = Batch.from_data_list(data_list[batch_id*500 : (batch_id+1)*500])
        batch = batch.to('cuda')
        confs = recon3d(batch.d_target, batch.edge_index, init_pos=batch.init_pos)[0][-1]
        for conf_id in range(500):
            conf_numpy = confs[conf_id : conf_id+num_nodes]
            conf_numpy = conf_numpy - conf_numpy.mean(axis=0)
            tmp_mol = copy.deepcopy(mol)
            rand_mol = set_rdmol_positions(tmp_mol,conf_numpy)
            smis2rands[smi].append(rand_mol)
    write_sdf(smis2rands[smi],osp.join(small_molecules_base,f'{smi_id}_nonpara.sdf'))
    write_sdf(smis2mols[smi],osp.join(small_molecules_base,f'{smi_id}_geom.sdf'))


# compute the rmsd matrix
small_molecules_base = './GEOM_drugs'
rmsd_allmat_min = []
for sdf_idx in tqdm(range(200)):
    ori_mols = read_sdf(osp.join(small_molecules_base,f'{sdf_idx}_geom.sdf'))
    num_ori = len(ori_mols)
    split = int(20/6*num_ori)
    all_gen_mols = []
    rand_pool = read_sdf(osp.join(small_molecules_base,f'{sdf_idx}_nonpara.sdf'))
    nopara_mols = rand_pool[:4*split]
    bond_mols = single_conf_gen_bonds(ori_mols[0],num_confs=split)
    opt_mols = rand_pool[4*split:5*split]
    opt_mols = [optimize_mol(i) for i in opt_mols]
    all_gen_mols = nopara_mols+bond_mols+opt_mols
    write_sdf(all_gen_mols,read_sdf(osp.join(small_molecules_base,f'{sdf_idx}_allgen.sdf')))

    coord_list = align_clustering(all_gen_mols,N=2*num_ori)
    cluster_mols = []
    for coord in coord_list:
        tmp_mol = copy.deepcopy(all_gen_mols[-1])
        cluster_mol = set_rdmol_positions(tmp_mol,coord)
        cluster_mols.append(cluster_mol)
    
    rmsd_mat = get_rmsd_mat(all_gen_mols,ori_mols)
    rmsd_allmat_min.append(rmsd_mat.min(axis=-1))
write_pkl(rmsd_allmat_min,'drugs_all_gen_rmsd_mat_min.pkl')