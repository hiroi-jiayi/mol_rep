import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np 
import plotly.express as px 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy import stats
import webbrowser

def align_SMILES(smiles, randomSeed = 0xf00d):
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(molecule, randomSeed = randomSeed)
    AllChem.MMFFOptimizeMolecule(molecule)#optimise 

    atoms_list = []
    for atom in molecule.GetAtoms():
        atoms_list.append(atom.GetSymbol())

    data = np.vstack((molecule.GetConformers()[0].GetPositions().T, np.array(atoms_list))).T
    molecule_df = pd.DataFrame(molecule.GetConformers()[0].GetPositions(), columns = ["x", "y", "z"]).astype(float)
    
    molecule_df.x = molecule_df.x - molecule_df.x.mean()
    molecule_df.y = molecule_df.y - molecule_df.y.mean()
    molecule_df.z = molecule_df.z - molecule_df.z.mean()
    
    pca = PCA(n_components=3)
    aligned_df = pd.DataFrame(pca.fit_transform(molecule_df), columns = ["x", "y", "z"])
    aligned_df["atom"] = atoms_list
    
    # shift everything to positive axis, find minimum coordinates of x, y, z to determine shift 
    x_min, y_min, z_min = min(aligned_df['x']), min(aligned_df['y']),min(aligned_df['z'])
    aligned_df['x'] = aligned_df['x'] + abs(x_min)
    aligned_df['y'] = aligned_df['y'] + abs(y_min)
    aligned_df['z'] = aligned_df['z'] + abs(z_min)

    divide = 5 # expand the grid 
    aligned_df['x_idx'] = (aligned_df['x']*divide).astype(int)
    aligned_df['y_idx'] = (aligned_df['y']*divide).astype(int)
    aligned_df['z_idx'] = (aligned_df['z']*divide).astype(int)
    
    return aligned_df

def create_grid(aligned_df):
    divide = 5
    x_max, y_max, z_max = max(aligned_df['x']), max(aligned_df['y']),max(aligned_df['z'])
    x_bound, y_bound, z_bound = int(x_max)+1,int(y_max)+1,int(z_max)+1 #define grid boundary 
    grid = np.zeros((x_bound*divide, y_bound*divide, z_bound*divide))
    
    for i in range(len(aligned_df)):
        x_target = aligned_df.loc[i,'x_idx']
        y_target = aligned_df.loc[i,'y_idx']
        z_target = aligned_df.loc[i,'z_idx']
        if aligned_df.loc[i,'atom'] == 'C':
            #grid[conformers_dict[count].loc[i,'x_idx']][conformers_dict[count].loc[i,'y_idx']][conformers_dict[count].loc[i,'z_idx']] = 6
            grid[x_target-7:x_target+7,y_target-7:y_target+7,z_target-7:z_target+7] = 12

        if aligned_df.loc[i,'atom'] == 'O':
            #grid[conformers_dict[count].loc[i,'x_idx']][conformers_dict[count].loc[i,'y_idx']][conformers_dict[count].loc[i,'z_idx']] = 8
            grid[x_target-6:x_target+6,y_target-6:y_target+6,z_target-6:z_target+6] = 16
                
        if aligned_df.loc[i,'atom'] == 'N':
            #grid[conformers_dict[count].loc[i,'x_idx']][conformers_dict[count].loc[i,'y_idx']][conformers_dict[count].loc[i,'z_idx']] = 7
            grid[x_target-6:x_target+7,y_target-6:y_target+7,z_target-6:z_target+7] = 14
                
        if aligned_df.loc[i,'atom'] == 'H':
            #grid[conformers_dict[count].loc[i,'x_idx']][conformers_dict[count].loc[i,'y_idx']][conformers_dict[count].loc[i,'z_idx']] = 1
            grid[x_target-2:x_target+3,y_target-2:y_target+3,z_target-2:z_target+3] = 1
                
        if aligned_df.loc[i,'atom'] == 'S':
            #grid[conformers_dict[count].loc[i,'x_idx']][conformers_dict[count].loc[i,'y_idx']][conformers_dict[count].loc[i,'z_idx']] = 16
            grid[x_target-10:x_target+10,y_target-10:y_target+10,z_target-10:z_target+10] = 32
            
        if aligned_df.loc[i,'atom'] == 'F':
            #grid[conformers_dict[count].loc[i,'x_idx']][conformers_dict[count].loc[i,'y_idx']][conformers_dict[count].loc[i,'z_idx']] = 9
            grid[x_target-5:x_target+5,y_target-5:y_target+5,z_target-5:z_target+5] = 19
        
        if aligned_df.loc[i,'atom'] == 'Br':
            #grid[conformers_dict[count].loc[i,'x_idx']][conformers_dict[count].loc[i,'y_idx']][conformers_dict[count].loc[i,'z_idx']] = 9
            grid[x_target-11:x_target+12,y_target-11:y_target+12,z_target-11:z_target+12] = 80
        
        if aligned_df.loc[i,'atom'] == 'Cl':
            #grid[conformers_dict[count].loc[i,'x_idx']][conformers_dict[count].loc[i,'y_idx']][conformers_dict[count].loc[i,'z_idx']] = 9
            grid[x_target-10:x_target+10,y_target-10:y_target+10,z_target-10:z_target+10] = 35
            
    return grid

def flatten(all_3D_grid):

    max_x = 0
    max_y = 0
    max_z = 0
    
    for i in range(len(all_3D_grid)):

        if all_3D_grid[i].shape[0] > max_x:
            max_x = all_3D_grid[i].shape[0]
        if all_3D_grid[i].shape[1] > max_y:
            max_y = all_3D_grid[i].shape[1]
        if all_3D_grid[i].shape[2] > max_z:
            max_z = all_3D_grid[i].shape[2]

    copy = all_3D_grid

    for i in range(len(copy)):
        x_pad = max_x - copy[i].shape[0]
        y_pad = max_y - copy[i].shape[1]
        z_pad = max_z - copy[i].shape[2]
        # padding symmetrically on both sides 
        copy[i] = np.pad(copy[i],((0,x_pad),(0,y_pad),(0,z_pad)), 'constant')
        copy[i] = copy[i].reshape(max_x*max_y*max_z)

    converted_array = []
    for i in range(len(copy)):
        converted_array.append(copy[i])

    converted_array = np.array(converted_array)
    converted_df = pd.DataFrame(converted_array)
    thres_30_df = converted_df
    col_var = converted_df.var()
    
    thres_30 = 0
    thres_30_idx = []
    for count, value in enumerate(col_var):
        if value > 30:
            thres_30 += 1
            thres_30_idx.append(count)
    
    print('Number of column with variance greater than 30: ' + str(thres_30))
    
    thres_30_df = converted_df.iloc[:,thres_30_idx]
            
    return thres_30_df

def rfr(n_estimators,x_train,x_test,y_train,y_test):

    clf = RandomForestRegressor(n_estimators = n_estimators)
    clf = clf.fit(x_train, y_train)

    y_forest_reg_pred = clf.predict(x_test)

    plt.plot([2, 11], [2, 11], "red")
    plt.scatter(y_test, y_forest_reg_pred)
    plt.title(str(j)+", filename: "+files[j]+
              "\nMSE: "+str(mean_squared_error(y_test, y_forest_reg_pred))+
              "\nr2: "+str(r2_score(y_test, y_forest_reg_pred))+
              "\nspear: "+str(stats.spearmanr(y_test, y_forest_reg_pred))
             )
    plt.xlabel("actual Y value")
    plt.ylabel("predicted Y value")
    plt.xlim(2, 11)
    plt.ylim(2, 11)

    plt.savefig("model_run_"+str(j)+".jpeg", bbox_inches = "tight", dpi =300)

    print("filename: "+files[j]+"\nMSE: "+str(mean_squared_error(y_test, y_forest_reg_pred))+"\nr2: "+str(r2_score(y_test, y_forest_reg_pred))+"\nspear: "+str(stats.spearmanr(y_test, y_forest_reg_pred)))
    
    return

j = int(sys.argv[1])

print("loaded "+str(j))

files = pd.read_csv('random_sel_files.csv')["file_name"].values

SMILES_df = {}

for obj in files:
    SMILES_df[obj] = pd.read_csv(os.getcwd()+"/dict_of_SMILES/" + obj)      
    SMILES_df[obj] = SMILES_df[obj].dropna()

print("finished filling SMILES_df")
    
all_aligned_df = {}
error = []
all_aligned_df[files[j]] = {}
for i in range(len(SMILES_df[files[j]])):
    try:
        aligned_df = align_SMILES(SMILES_df[files[j]]['SMILES'][i],9)
        all_aligned_df[files[j]][i] = aligned_df
        
    except:
        error = int(i)
        SMILES_df[files[j]] = SMILES_df[files[j]].drop(error)

print("finished filling all_aligned_df")
        
for i in range(len(files)):
    SMILES_df[files[i]] = SMILES_df[files[i]].reset_index()

    
all_3D_grid = {}
all_3D_grid[files[j]] = {}

for i in range(int(SMILES_df[files[j]]['index'][-1:])+1):
    try:
        grid = create_grid(all_aligned_df[files[j]][i])
        all_3D_grid[files[j]][i] = grid

    except:
        all_3D_grid[files[j]][i] = np.zeros((0,0,0))
    
print("finished filling all_3D_grid")

flattened_df = {}
flattened_df[files[j]] = flatten(all_3D_grid[files[j]])
flattened_df[files[j]] = flattened_df[files[j]].loc[~(flattened_df[files[j]]==0).all(axis=1)].reset_index()


x_train_0, x_test_0, y_train_0, y_test_0 = train_test_split(flattened_df[files[j]], SMILES_df[files[j]]['pXC50'], test_size = 0.2)


rfr(100, x_train_0, x_test_0, y_train_0, y_test_0)

webbrowser.open('https://youtu.be/PV-UYFTitFUA')

print("finished everything")
