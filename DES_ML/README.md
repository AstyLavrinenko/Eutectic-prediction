# Prerequisites
* Python 3.7.15
# Requirements
* `conda config --append channels conda-forge`
* `conda install --file requirements.txt`
# Melting temperature prediction

Deep eutectic solvents (DESs) represent an environmental-friendly alternative to the conventional organic solvents. The application area of DESs is determined by their liquid range, thus, the prediction of solid-liquid equilibrium (SLE) diagram is essential for the development of new DESs. Herein, we present machine learning model based on support vector regression for computing melting points of DESs. 

The data that support the findings of this study are openly available in this repository https://github.com/AstyLavrinenko/Eutectic-prediction/tree/main/DES_ML
There are dataset, generated descriptors, feature selection and optimisation algorithms, and developed machine learning model which were used in the project.

Such descriptors as sigma_moments, sigma_profiles, and infinite_dilution were calculated using COSMOtherm software.

# Citation
A.K. Lavrinenko, I.Yu. Chernyshov, N.S. Serov, Machine Learning Approach for the Prediction of Eutectic Temperatures for Metal-Free Deep Eutectic Solvents
