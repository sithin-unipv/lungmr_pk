# Non-Invasive Programmed Death-Ligand (PD-L1) Stratification in Non-Small Cell Lung Cancer using Dynamic Contrast-Enhanced MRI (DCE-MRI)

Original repository supporting the article submitted to European Radiology (@Citation TBA)

### **Repository structure**
#### **Overview:**

* Automated Bolus Arrival Time (BAT) estimation methods - LinearLinear, LinearQuadratic, and PeakGradient
* Population Arterial Input Functions (AIF) - Weinmann, Parker and Georgiou
* Fit Quality Analysis
* Pharacokinetic (PK) Tofts Models - Standard and Extended Tofts
* Statistical Analysis
* Data Analysis and Visualizations
* Jupyter notebooks with usage examples

#### **Contents:**
```
lungmr_pk
├── qiba                               # directory copied from https://github.com/OSIPI/DCE-DSC-MRI_CodeCollection/tree/develop/src/original/LCB_BNI_USA for sanity checks
│    └── data.mat   
│    └── dce.py
│    └── dsc.py

├── pkutils.py                         # Python file with all the APIs for PK modeling in DCE MR sequence with population AIF and automated BAT estimation methods
                                       # All the APIs or functions can be assessed by copying this file to your projects

├── data_analysis.ipynb                # Jupyter notebook used for gathering the image acquisition parameters
|── pk_modelling_sanity_check.ipynb    # Jupyter notebook used to check the correctness or sanity of the developed APIs using example data
|── pk_fit_quanlity_analysis.ipynb     # Jupyter notebook used for assessing the fit quality of various Tofts model design choices
|── pk_modelling.ipynb                 # Best fit Tofts model used for mean-tumor/voxel-wise modeling and statistical analysis
|── visualizations.ipynb               # Jupyter notebook used for generating distribution and ROC-AUC plots

├── LICENCE                            # GNU General Public License v3.0
├── README.md

```
