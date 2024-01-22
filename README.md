# DeepComBat
--------
**Maintainer**: Fengling Hu, fengling.hu@pennmedicine.upenn.edu

## Table of content
- [1. Installation](#id-section1)
- [2. Background](#id-section2)
- [3. Software](#id-section3)
- [4. Citation](#id-section4)

<div id='id-section1'/>

## 1. Installation
The R package can be installed via devtools by running the following code

```
# install.packages("devtools")
devtools::install_github("hufengling/DeepComBat")
```

Then, you can load this package via

```
library(DeepComBat)
```

## 2. Background
Neuroimaging data acquired using multiple scanners or protocols are increasingly available. However, such data exhibit technical artifacts across batches which introduce confounding and decrease reproducibility. This is especially true when multi-batch data are analyzed using complex downstream models which are more likely to pick up on and implicitly incorporate batch-related information. Previously-proposed image harmonization methods have sought to remove these batch effects; however, batch effects remain detectable in the data after applying these methods. We design DeepComBat to learn and remove multivariate batch effects and show it can almost fully prevent detection of scanner properties.

DeepComBat is meant to be applied after initial preprocessing of the images to obtain a set of features and before statistical analyses. The application of DeepComBat is not limited to neuroimaging data; however, it has yet to be tested in other types of data.

<div id='id-section3'/>

## 3. Software

The `DeepComBat` package provides four main functions for end-users.

`deepcombat_setup()` is used to set up inputs for DeepComBat as well as initialize the DeepComBat CVAE. `deepcombat_train()` provides functionality for training the DeepComBat CVAE from the initialized `deepcombat_setup()` object. Finally, `deepcombat_harmonize()` takes outputs from `deepcombat_setup()` and `deepcombat_train()` and allows users to pass data through the trained DeepComBat CVAE and harmonize the data.

If users seek to use DeepComBat for external harmonization (ie train DeepComBat on part of the data and use the trained model to harmonize other data), `deepcombat_setup_from_train()` can be used. Then, `deepcombat_harmonize()` takes outputs from `deepcombat_setup_from_train()`, `deepcombat_train()`, and `deepcombat_harmonize()` run on the training data. For external harmonization DeepComBat, all estimation is conducted on the training data, including standardization, CVAE parameters, latent space ComBat, and residual ComBat.

## 4. Citation
If you are using DeepComBat for harmonization, please cite the following article:

Hu, F., Lucas, A., Chen, A.A., Coleman, K., Horng, H., Ng, R.W., Tustison, N.J., Davis, K.A., Shou, H., Li, M., Shinohara, R., The Alzheimer's Disease Neuroimaging Initiative. 2023. DeepComBat: A Statistically Motivated, Hyperparameter-Robust, Deep Learning Approach to Harmonization of Neuroimaging Data. bioRxiv.
