# mRNA display/deep learning
 
## About

This repository holds python scripts to reproduce the results of mRNA display/deep learning-based studies to profile the substrate preferences of i) LazBF and LazDEF enzymes, and ii) the entire lactazole biosynthetic pathway (LazBCDEF). The code enables to reproduce the entire workflow from from loading NGS output files to training and evaluating tensorflow-based models.

1. All metaparametes are specified in code/config.py (set up to analyze library 6C6, i.e., LazDEF profiling results)
2. Primary code is in ./code/utils
3. Model definition can be found in ./code/tf/cnn_model.py (cnn_v5 was used to build LazBF and LazDEF models; cnn_v6 - LazBCDEF)
5. Fully trained model weights are either ./model/55_r6_cnn_v5_fully_trained.h5 (LazBF model); model/66_r5_ccn_v5_fully_trained_v2.h5 (LazDEF); or ./model/full_pathway_DADL_811_fully_trained.h5 (LazBCDEF)
6. ECFP feature matrices are ./feature_matrices/DENSE_Morgan_F_r=4_LazBF (LazBF), ./feature_matrices/DENSE_Morgan_F_r=4_LazDEF (LazDEF) and ./feature_matrices/DENSE_Morgan_F_r=4_JS.npy (LazBCDEF)
7. The associated NGS sequencing data is uploaded to DDBJ (accession number: DRA013287 for LazBF and LazDEF; DRA016846 for LazBCDEF)
\
\
\
For further details, refer to the publications: [LazBF and LazDEF](https://pubs.acs.org/doi/10.1021/acscentsci.2c00223), and [LazBCDEF](https://pubs.acs.org/doi/10.1021/acscentsci.3c00957).
Please cite A. Vinogradov _et al. ACS Cent. Sci._ __2022__, 8, 6, 814–824 and J. Chang _et al. ACS Cent. Sci._ __2023__, doi: 10.1021/acscentsci.3c00957 if you use this code.

## Dependencies

The scripts were written and tested for 

python 3.8.5 \
numpy 1.19.5 \
pandas 1.2.4 \
rdkit 2021.03.3 \
tensorflow 2.4.1 \
tensorflow-io 0.17.1 \
h5py 2.10.0 \
matplotlib 3.3.4 \
seaborn 0.11.1

## Usage examples

An example of building a LazDEF substrate preference model from NGS data can be found in a jupyter notebook ./code/LazDEF_model_training_walkthrough.ipynb

## License

_The code is released under the GNU General Public License v3.0 without any explicit or implied warranty. Use it at your own discretion._
