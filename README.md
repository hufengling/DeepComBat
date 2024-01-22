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

Example code is provided below:
```
setup <- deepcombat_setup(~ age + sex + diagnosis, 
                          ~ scanner, 
                          data, 
                          covariates) # data frame with columns: age, sex, diagnosis, scanner
trained_model <- deepcombat_trainer(setup, verbose = TRUE)
harmonized <- deepcombat_harmonize(setup, trained_model)
```

If users seek to use DeepComBat for external harmonization (ie train DeepComBat on part of the data and use the trained model to harmonize other data), `deepcombat_setup_from_train()` can be used. Then, `deepcombat_harmonize()` takes outputs from `deepcombat_setup_from_train()`, `deepcombat_train()`, and `deepcombat_harmonize()` run on the training data. For external harmonization DeepComBat, all estimation is conducted on the training data, including standardization, CVAE parameters, latent space ComBat, and residual ComBat.

If external harmonization is desired, we recommend using a regularized version of DeepComBat to improve out-of-sample performance. This can be accomplished by setting the `use_default_optim` parameter in `deepcombat_setup()` to "FALSE" and using the AdamW optimizer instead of the Adam optimizer (default) -- to do so, set the `optimizer` parameter in `deepcombat_trainer()` to `optim_adamw(setup_train$cvae$parameters, lr = 0.01))`

```
# Train model using training data
setup_train <- deepcombat_setup(~ age + sex + diagnosis, 
                          ~ scanner, 
                          data_train, 
                          covariates_train, # data frame with columns: age, sex, diagnosis, scanner
                          use_default_optim = FALSE)
trained_model <- deepcombat_trainer(setup_train, # Train DeepComBat CVAE using training data
                                    verbose = TRUE, 
                                    optimizer = optim_adamw(setup_train$cvae$parameters, lr = 0.01))
harmonized_train <- deepcombat_harmonize(setup_train, 
                                         trained_model) # Estimate latent space and residual ComBat

# Apply trained model to external data
setup_external <- deepcombat_setup_from_train(setup_train, 
                                              ~ age + sex + diagnosis, 
                                              ~ scanner, 
                                              data_external, 
                                              covariates_external) # Set up input matrices
harmonized_external <- deepcombat_harmonize(setup_external, 
                                            trained_model, 
                                            harmonized_train, 
                                            verbose = TRUE) # Pass external data through trained DeepComBat
```

## 4. Citation
If you are using DeepComBat for harmonization, please cite the following article:

Hu, F., Lucas, A., Chen, A.A., Coleman, K., Horng, H., Ng, R.W., Tustison, N.J., Davis, K.A., Shou, H., Li, M., Shinohara, R., The Alzheimer's Disease Neuroimaging Initiative. 2023. DeepComBat: A Statistically Motivated, Hyperparameter-Robust, Deep Learning Approach to Harmonization of Neuroimaging Data. bioRxiv.
