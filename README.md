# mol_rep
Chemical representations of molecules (fingerprints and voxel-based approach) combined with random forest regressor for predicting drug potency (pXC50)

Several molecular representations were explored here, including fingerprints (Morgan, RDKit, MACCS, phamacophore) and voxel-based representations. The task was to compare the predictive ability of these representations in drug discovery appliations. Dataset used in this project was described in https://link.springer.com/article/10.1007/s10994-017-5685-x under Section: Baseline QSAR datasets. The quantity of interest here is the pXC50 value.

Out of 2094 datasets in data folder, 200 were randomly selected. The datasets used for experiments were saved in 'random_sel_files.csv'. There are three stages in this project:

(1) generation of fingerprints, which mainly relied on RDKit package;
(2) data-preprocessing and application of random forest regressor (n-estimator = 1000, CV = 5);
(3) result analysis, shown in the folder 'result_analysis'. 

All machine learning scipt for Morgan fingerprints and MACCS fingerprints were shown. For other fingerprint representation, one example python file and a slurm file (.sh) were shown. Due to the large computational power required, jobs were submitted to the Cambridge Service for Data-Driven Discovery (CSD3). 

