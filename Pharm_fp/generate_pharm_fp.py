from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import AllChem
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
import pandas as pd
import numpy as np
 
        
files = pd.read_csv('random_sel_files.csv')["file_name"].values

smiles_from_file = {}

for obj in files:
    smiles_from_file[obj] = pd.read_csv("dict_of_SMILES/" + obj)

file_name = [name for name in smiles_from_file]


def pharma(SMILES):
    mol = Chem.MolFromSmiles(SMILES)
    AllChem.EmbedMolecule(mol) #gen 3d
    fp_3d = Generate.Gen2DFingerprint(mol,Gobbi_Pharm2D.factory, dMat = Chem.Get3DDistanceMatrix(mol))
    vector_3d = np.array(fp_3d)
    return vector_3d

smiles = {}

for i in range(70,110):
    all_pharma = []
    print(file_name[i])

    for j in range(len(smiles_from_file[file_name[i]])):
        #print(smiles_from_file[file_name[i]].loc[j,'SMILES'])
        try:
            fp_pharma = pharma(smiles_from_file[file_name[i]].loc[j,'SMILES'])
            all_pharma.append(fp_pharma)
            print(j)
        except:
            print('Error at'+str(j))
            
            
    fp_pharma = pd.DataFrame(all_pharma)

    smiles[file_name[i]] = pd.concat([smiles_from_file[file_name[i]], fp_pharma], axis=1)
    smiles[file_name[i]].to_csv('pharma3D/'+str(file_name[i]), index=False)

    print(file_name[i]+' completed')  