import glob
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import DataStructs
from rdkit.Chem import AllChem
import numpy as np

smiles_files = glob.glob("dict_of_SMILES/*.csv")

smiles_from_file = {}

# Generate dictionary of dataframe
for count, i in enumerate(smiles_files):
    smiles_from_file[i.replace('dict_of_SMILES/','')] = pd.read_csv(i)

file_name = [name for name in smiles_from_file]

def MorganFP_rad(SMILES, radius):
    mol = Chem.MolFromSmiles(SMILES)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=1024)
    array = np.zeros((0, ), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, array)
    vector = np.array(fp)
    return vector 

smiles_4 = {}
smiles_3 = {}
smiles_2 = {}
smiles_5 = {}

for i in range(len(file_name)):
    all_morganfp_rad4 = []
    all_morganfp_rad3 = []
    all_morganfp_rad2 = []
    all_morganfp_rad5 = []

    for j in range(len(smiles_from_file[file_name[i]])):
        try:
            fp4 = MorganFP_rad(smiles_from_file[file_name[i]].loc[j,'SMILES'], 4)
            fp2 = MorganFP_rad(smiles_from_file[file_name[i]].loc[j,'SMILES'], 2)
            fp3 = MorganFP_rad(smiles_from_file[file_name[i]].loc[j,'SMILES'], 3)
            fp5 = MorganFP_rad(smiles_from_file[file_name[i]].loc[j,'SMILES'], 5)

            all_morganfp_rad4.append(fp4)
            all_morganfp_rad3.append(fp3)
            all_morganfp_rad2.append(fp2)
            all_morganfp_rad5.append(fp5)
        except:
            print('Error at'+str(j))
            
    fp_df4 = pd.DataFrame(all_morganfp_rad4)
    fp_df3 = pd.DataFrame(all_morganfp_rad3)
    fp_df2 = pd.DataFrame(all_morganfp_rad2)
    fp_df5 = pd.DataFrame(all_morganfp_rad5)

    smiles_4[file_name[i]] = pd.concat([smiles_from_file[file_name[i]], fp_df4], axis=1)
    smiles_3[file_name[i]] = pd.concat([smiles_from_file[file_name[i]], fp_df3], axis=1)
    smiles_2[file_name[i]] = pd.concat([smiles_from_file[file_name[i]], fp_df2], axis=1)
    smiles_5[file_name[i]] = pd.concat([smiles_from_file[file_name[i]], fp_df5], axis=1)

    smiles_4[file_name[i]].to_csv('Morgan_fp_rad4/'+str(file_name[i]), index=False)
    smiles_3[file_name[i]].to_csv('Morgan_fp_rad3/'+str(file_name[i]), index=False)
    smiles_2[file_name[i]].to_csv('Morgan_fp_rad2/'+str(file_name[i]), index=False)
    smiles_5[file_name[i]].to_csv('Morgan_fp_rad5/'+str(file_name[i]), index=False)

    print(file_name[i]+' completed')