# infinite-physical-monkey
Do Deep Learning Methods Really Perform Better in Binding Pose Generation?

<div align=center>
<img src="./pic/inf_phy_monkey.png" width="50%" height="50%" alt="TOC" align=center />
</div>
#### infinite physical monkey for molecular conformation generation

Generate non-parameter molecules with 

```python
python inp_phy_mon_MCG.py
```

The dataset could be downloaded [here](https://drive.google.com/drive/folders/10dWaj5lyMY0VY4Zl0zDPCa69cuQUGb-6?usp=sharing). Or prepare from GEOM dataset locally, details could be found at [ConfGF](https://github.com/DeepGraphLearning/ConfGF). 

#### infinite physical monkey for molecular conformation generation

Download the PDBBind Core dataset first, and unzip it as 'core'

Generate rdkit mols first

```python
python rdkit_mols.py
```

Align the rdkit molecules to the center of the pocket and perform random rotation on them. Then use Vina to assess the binding energy between infinite physical monkey conformations between pockets and ligands. 

```python
python scoring_random.py
```

Finally, perform the force field optimization for the molecules inside rigid pockets. 

```
python uff_opt.py
```

