#' Create model matrices for batch correction
#'
#' This function takes in a formula for covariates, a formula for batch, a feature matrix,
#' and a covariate matrix and generates the corresponding model matrices for these variables, excluding intercept.
#' It also generates a target batch model matrix that will be used to adjust the batch effect in the feature matrix.
#'
#' @param covariate_formula formula specifying the covariates to preserve. Ex: ~ Covariate1 + Covariate2
#' @param batch_formula formula specifying the batch variable to correct for. Ex: ~ Batch
#' @param feature_matrix a matrix or dataframe of features with rows representing subjects and columns representing features.
#' @param covariate_matrix a matrix or dataframe of covariates with rows representing subjects and columns representing covariates. Columns must be named such that covariate_formula can extract the correct columns.
#' @param reference_batch the name of the batch to be used as the reference level for batch correction. If not specified,
#' the function will generate a target batch that maps to an intermediate space between all batches. Intermediate-space harmonization is currently only implemented for correction of two batches.
#'
#' @return a list with the following components:
#' \describe{
#' \item{features}{a scaled feature matrix}
#' \item{covariates}{the model matrix for covariates, excluding intercept, with each column scaled to have values between 0 and 1.}
#' \item{batch}{the model matrix for batch, excluding intercept}
#' \item{target_batch}{the model matrix for the target batch, which will be used to adjust the batch effect in the feature matrix}
#' }
#'
#' @import stats
#' @importFrom dplyr pull
#'
#' @examples
#'\dontrun{
#' create_model_matrices(covariate_formula = ~ age + sex + diagnosis,
#' batch_formula = ~ batch,
#' feature_matrix = data, covariate_matrix = covariates,
#' reference_batch = "batch1")
#' }
create_model_matrices <- function(covariate_formula, batch_formula,
                                  feature_matrix, covariate_matrix,
                                  reference_batch = NULL) {

  # Check conditions
  stopifnot("'feature_matrix' and 'covariate_matrix' should have the same number of rows. Rows represent subjects and columns represent features/covariates."=
              nrow(feature_matrix) == nrow(covariate_matrix))
  stopifnot("'batch_formula' can only have one variable."=
              !grepl(" ", deparse(batch_formula)))
  stopifnot("If reference batch is desired, 'reference_batch' must match one of the batch names in the 'covariate_matrix' column referred to by 'batch_formula'"=
              is.null(reference_batch) | any(covariate_matrix[batch_formula[[2]]] == reference_batch))

  # Helper function to adjust range of each column to be between 0 and 1
  range01 <- function (df) {
    df01 <- apply(df, 2, function (col) {
      (col - min(col)) / (max(col) - min(col))
    })
    attr(df01, "minmax") <- apply(df, 2, function (col) {
      c(min(col), max(col))
    })
    return(df01)
  }

  # Generate model matrix objects of covariates and batch, excluding intercept
  covariate_mm <- model.matrix(covariate_formula, data = covariate_matrix)[, -1] # Remove intercept
  covariate_mm_01 <- range01(covariate_mm)
  batch_mm <- as.matrix(model.matrix(batch_formula, data = covariate_matrix)[, -1]) # Remove intercept and make sure batch_mm is always a 2D matrix (even if only one column)
  attr(batch_mm, "batch_names") <- unique(pull(covariate_matrix, batch_formula[[2]]))
  if (is.null(reference_batch)) {
    # Maps to some intermediate space where the "contrapositive" is something similar to all batches
    if(is.null(ncol(batch_mm))) {
      target_batch_mm <- matrix(rep(0.5, length(batch_mm)), ncol = 1)
    } else {
      target_batch_mm <- matrix(rep(1 / (ncol(batch_mm) + 1), length(batch_mm)), ncol = ncol(batch_mm))
    }
  } else {
    # Need to find one row in covariate_matrix$batch that corresponds to reference batch so that we can copy its model.matrix representation
    reference_batch_ind <- which(covariate_matrix[batch_formula[[2]]] == reference_batch)[1]
    target_batch_mm <- matrix(rep(batch_mm[reference_batch_ind, ], nrow(batch_mm)),
                              ncol = ncol(batch_mm), byrow = TRUE)
  }

  return(list(features = scale(feature_matrix),
              covariates = covariate_mm_01,
              batch = batch_mm,
              target_batch = target_batch_mm))
}

#' Create model matrices for batch correction from trained model
#'
#' This function takes in a formula for covariates, a formula for batch, a feature matrix,
#' and a covariate matrix and generates the corresponding model matrices for these variables, excluding intercept.
#' It also generates a target batch model matrix that will be used to adjust the batch effect in the feature matrix.
#'
#' @param setup_input output from deepcombat_setup containing training estimands and settings
#' @param covariate_formula formula specifying the covariates to preserve. Ex: ~ Covariate1 + Covariate2
#' @param batch_formula formula specifying the batch variable to correct for. Ex: ~ Batch
#' @param feature_matrix a matrix or dataframe of features with rows representing subjects and columns representing features.
#' @param covariate_matrix a matrix or dataframe of covariates with rows representing subjects and columns representing covariates. Columns must be named such that covariate_formula can extract the correct columns.
#' @param reference_batch the name of the batch to be used as the reference level for batch correction. If not specified,
#' the function will generate a target batch that maps to an intermediate space between all batches. Intermediate-space harmonization is currently only implemented for correction of two batches.
#'
#' @return a list with the following components:
#' \describe{
#' \item{features}{a scaled feature matrix}
#' \item{covariates}{the model matrix for covariates, excluding intercept, with each column scaled to have values between 0 and 1.}
#' \item{batch}{the model matrix for batch, excluding intercept}
#' \item{target_batch}{the model matrix for the target batch, which will be used to adjust the batch effect in the feature matrix}
#' }
#'
#' @import stats
#' @importFrom dplyr pull
#'
#' @examples
#'\dontrun{
#' create_model_matrices_from_train(setup_input, covariate_formula = ~ age + sex + diagnosis,
#' batch_formula = ~ batch,
#' feature_matrix = data, covariate_matrix = covariates,
#' reference_batch = "batch1")
#' }
create_model_matrices_from_train <- function(setup_input, covariate_formula, batch_formula,
                                             feature_matrix, covariate_matrix,
                                             reference_batch = NULL) {

  # Check conditions
  stopifnot("'feature_matrix' and 'covariate_matrix' should have the same number of rows. Rows represent subjects and columns represent features/covariates."=
              nrow(feature_matrix) == nrow(covariate_matrix))
  stopifnot("'batch_formula' can only have one variable."=
              !grepl(" ", deparse(batch_formula)))
  stopifnot("If reference batch is desired, 'reference_batch' must match one of the batch names in the 'covariate_matrix' column referred to by 'batch_formula'"=
              is.null(reference_batch) | any(covariate_matrix[, as.character(batch_formula[[2]])] == reference_batch))

  # Helper function to adjust range of each column to be between 0 and 1
  range01_from_train <- function (df, setup_input) {
    df01 <- df
    minmax <- attr(setup_input$covariates, "minmax")
    for (i in 1:ncol(df)) {
      df01[, i] <- (df[, i] - minmax[1, i]) / (minmax[2, i] - minmax[1, i])
    }
    attr(df01, "minmax") <- minmax
    return(df01)
  }

  # Generate model matrix objects of covariates and batch, excluding intercept
  covariate_mm <- model.matrix(covariate_formula, data = covariate_matrix)[, -1] # Remove intercept
  covariate_mm_01 <- range01_from_train(covariate_mm, setup_input)
  covariate_matrix[, as.character(batch_formula[[2]])] <- factor(pull(covariate_matrix, batch_formula[[2]]), levels = attr(setup_input$batch, "batch_names"))
  batch_mm <- as.matrix(model.matrix(batch_formula, data = covariate_matrix)[, -1]) # Remove intercept and make sure batch_mm is always a 2D matrix (even if only one column)
  attr(batch_mm, "batch_names") <- attr(setup_input$batch, "batch_names")
  if (is.null(reference_batch)) {
    # Maps to some intermediate space where the "contrapositive" is something similar to all batches
    if(is.null(ncol(batch_mm))) {
      target_batch_mm <- matrix(rep(0.5, length(batch_mm)), ncol = 1)
    } else {
      target_batch_mm <- matrix(rep(1 / (ncol(batch_mm) + 1), length(batch_mm)), ncol = ncol(batch_mm))
    }
  } else {
    # Need to find one row in covariate_matrix$batch that corresponds to reference batch so that we can copy its model.matrix representation
    reference_batch_ind <- which(attr(setup_input$batch, "batch_names") == reference_batch)
    target_batch_mm <- matrix(0, nrow = nrow(batch_mm), ncol = ncol(batch_mm))
    if (reference_batch_ind != 1) {
      target_batch_mm[, reference_batch_ind - 1] <- 1
    }
  }
  center_train <- attr(setup_input$features, "scaled:center")
  scale_train <- attr(setup_input$features, "scaled:scale")
  feature_matrix_scale <- (feature_matrix - matrix(center_train,
                                                   nrow = nrow(feature_matrix),
                                                   ncol = ncol(feature_matrix),
                                                   byrow = T)) /
    matrix(scale_train,
           nrow = nrow(feature_matrix),
           ncol = ncol(feature_matrix),
           byrow = T)
  attr(feature_matrix_scale, "scaled:center") <- center_train
  attr(feature_matrix_scale, "scaled:scale") <- scale_train

  return(list(features = feature_matrix_scale,
              covariates = covariate_mm_01,
              batch = batch_mm,
              target_batch = target_batch_mm))
}

deepcombat_dataset <- dataset(
  name = "deepcombat_dataloader",

  initialize = function(input_list) {
    self$features <- torch_tensor(input_list$features)
    self$covariates <- torch_tensor(input_list$covariates)
    self$batch <- torch_tensor(input_list$batch)
    self$target_batch <- torch_tensor(input_list$target_batch)
  },
  .getitem = function(index) {
    features <- self$features[index, ]
    covariates <- self$covariates[index, ]
    batch <- self$batch[index]
    target_batch <- self$target_batch[index]
    return(list(features,
                covariates,
                batch,
                target_batch))
  },
  .length = function() {
    self$features$size()[[1]]
  }
)

#' DeepComBat Setup Helper
#'
#' Sets up the DeepComBat analysis by creating a torch dataloader, CVAE architecture and torch optimizer.
#' This function takes in a formula for covariates, a formula for batch, a feature matrix,
#' and a covariate matrix and generates the corresponding model matrices for these variables, excluding intercept.
#' It also generates a target batch model matrix that will be used to adjust the batch effect in the feature matrix.
#'
#' @param covariate_formula formula specifying the covariates to preserve. Ex: ~ Covariate1 + Covariate2
#' @param batch_formula formula specifying the batch variable to correct for. Ex: ~ Batch
#' @param feature_matrix a matrix or dataframe of features with rows representing subjects and columns representing features.
#' @param covariate_matrix a matrix or dataframe of covariates with rows representing subjects and columns representing covariates. Columns must be named such that covariate_formula can extract the correct columns.
#' @param reference_batch the name of the batch to be used as the reference level for batch correction. If not specified, the function will generate a target batch that maps to an intermediate space between all batches. Intermediate-space harmonization is currently only implemented for correction of two batches.
#' @param cvae_settings Either a character string "default" or a named list with items "batch_size", "latent_dim", "n_hidden" and/or "vae_dim". Default settings use batch_size of 64, latent_dim that is the closest power of 2 to 1/4th of the number of features, and n_hidden of 3.
#' @param use_default_optim A logical indicating whether to use the default optimizer (TRUE) or not (FALSE). If TRUE, Adam optimizer with learning rate of 0.1 and no weight decay is used. If FALSE, optimizer will be set to NULL, and the desired optimizer can be passed to 'deepcombat_trainer' manually.
#'
#' @return An object of class "deepcombat_setup_object" containing the dataloader, CVAE architecture, and optimizer, and a copy of the normalized/standardized input data
#' @export
#'
#' @examples
#' \dontrun{
#' deepcombat_setup(covariate_formula = ~ age + sex + diagnosis,
#' batch_formula = ~ batch,
#' feature_matrix = data, covariate_matrix = covariates,
#' reference_batch = "batch1",
#' cvae_settings = list(batch_size = 64, latent_dim = 16, n_hidden = 2),
#' use_default_optim = TRUE)
#' }
#' @import torch
deepcombat_setup <- function(covariate_formula, batch_formula,
                             feature_matrix, covariate_matrix,
                             reference_batch = NULL,
                             cvae_settings = "default",
                             use_default_optim = TRUE) {
  # Condition checking and setting defaults
  stopifnot("'cvae_settings must be either a string or named list"= is.character(cvae_settings) | is.list(cvae_settings))
  if (is.character(cvae_settings)) {
    stopifnot("'cvae_settings' must be 'default' or named list with items 'batch_size', 'latent_dim' and/or 'n_hidden'"=
                cvae_settings == "default")
    batch_size <- 64
    n_hidden <- 3
    vae_dim <- NULL
  }
  if (is.list(cvae_settings)) {
    stopifnot("'cvae_settings' must be 'default' or named list with items 'batch_size', 'latent_dim' and/or 'n_hidden'"=
                "batch_size" %in% names(cvae_settings) | "latent_dim" %in% names(cvae_settings) | "n_hidden" %in% names(cvae_settings) | "vae_dim" %in% names(cvae_settings))
    if ("batch_size" %in% names(cvae_settings))
      batch_size <- cvae_settings$batch_size
    else
      batch_size <- 64

    if ("n_hidden" %in% names(cvae_settings))
      n_hidden <- cvae_settings$n_hidden
    else
      n_hidden = 3

    if ("vae_dim" %in% names(cvae_settings))
      vae_dim <- cvae_settings$vae_dim
    else
      vae_dim <- NULL
  }

  # Creating dataloader
  input_list <- create_model_matrices(covariate_formula, batch_formula,
                                      feature_matrix, covariate_matrix,
                                      reference_batch)
  dl <- dataloader(deepcombat_dataset(input_list),
                   batch_size = batch_size, shuffle = TRUE)

  # Creating CVAE architecture
  feature_dim <- ncol(input_list$features)
  n_batch <- ncol(input_list$batch)
  n_covariate <- ncol(input_list$covariates)

  if (is.character(cvae_settings)) # If it is a character vector, it should be 'default'
    latent_dim <- 2^round(log(feature_dim / 4, 2)) # Divide feature_dim by 4 and round to the nearest power of 2
  if (is.list(cvae_settings)) {
    if ("latent_dim" %in% names(cvae_settings))
      if ("vae_dim" %in% names(cvae_settings)) {
        warning("vae_dim and latent_dim are both provided. Using last element of vae_dim as latent_dim")
        latent_dim <- cvae_settings$vae_dim[length(cvae_settings$vae_dim)]
      } else {
        latent_dim <- cvae_settings$latent_dim
      }
    else
      latent_dim <- 2^round(log(feature_dim / 4, 2)) # Divide feature_dim by 4 and round to the nearest power of 2
  }

  cvae <- deepcombat_cvae(feature_dim, latent_dim,
                          n_hidden, n_batch, n_covariate,
                          vae_dim)

  # Creating optimizer
  if (use_default_optim) {
    optimizer <- optim_adam(cvae$parameters, lr = 0.01, weight_decay = 0)
  } else {
    optimizer <- NULL # If not using default, will have to pass to trainer manually
  }

  setup_object = list(dataloader = dl,
                      cvae = cvae,
                      optimizer = optimizer,
                      input = input_list,
                      cvae_settings = list(batch_size = batch_size,
                                           latent_dim = latent_dim,
                                           n_hidden = n_hidden,
                                           vae_dim = vae_dim))
  class(setup_object) = "deepcombat_setup_object"
  return(setup_object)
}

#' DeepComBat Setup Helper from Training
#'
#' Sets up the DeepComBat analysis by creating a torch dataloader, CVAE architecture and torch optimizer.
#' This function takes in a formula for covariates, a formula for batch, a feature matrix,
#' and a covariate matrix and generates the corresponding model matrices for these variables, excluding intercept.
#' It also generates a target batch model matrix that will be used to adjust the batch effect in the feature matrix.
#'
#' @param setup_train input list from deepcombat_setup run on training data. Ex: setup_obj$input
#' @param covariate_formula formula specifying the covariates to preserve. Ex: ~ Covariate1 + Covariate2
#' @param batch_formula formula specifying the batch variable to correct for. Ex: ~ Batch
#' @param feature_matrix a matrix or dataframe of features with rows representing subjects and columns representing features.
#' @param covariate_matrix a matrix or dataframe of covariates with rows representing subjects and columns representing covariates. Columns must be named such that covariate_formula can extract the correct columns.
#' @param reference_batch the name of the batch to be used as the reference level for batch correction. If not specified, the function will generate a target batch that maps to an intermediate space between all batches. Intermediate-space harmonization is currently only implemented for correction of two batches.
#'
#' @return An object of class "deepcombat_setup_object" containing the dataloader, CVAE architecture, and optimizer, and a copy of the normalized/standardized input data
#' @export
#'
#' @examples
#' \dontrun{
#' deepcombat_setup_from_train(covariate_formula = ~ age + sex + diagnosis,
#' batch_formula = ~ batch,
#' feature_matrix = data, covariate_matrix = covariates,
#' reference_batch = "batch1",
#' cvae_settings = list(batch_size = 64, latent_dim = 16, n_hidden = 2),
#' use_default_optim = TRUE)
#' }
#' @import torch
deepcombat_setup_from_train <- function(setup_train,
                                        covariate_formula,
                                        batch_formula,
                                        feature_matrix,
                                        covariate_matrix,
                                        reference_batch = NULL) {
  if (class(setup_train) != "deepcombat_setup_object") {
    stop("setup_train must be the output from deepcombat_setup run on the training data.")
  }
  #CVAE settings
  batch_size <- setup_train$cvae_settings$batch_size

  # Creating dataloader
  input_list <- create_model_matrices_from_train(setup_train$input, covariate_formula, batch_formula,
                                                 feature_matrix, covariate_matrix,
                                                 reference_batch)
  dl <- dataloader(deepcombat_dataset(input_list),
                   batch_size = batch_size, shuffle = TRUE)

  # # Creating CVAE architecture
  # feature_dim <- ncol(input_list$features)
  # n_batch <- ncol(input_list$batch)
  # n_covariate <- ncol(input_list$covariates)
  #
  # if (is.character(cvae_settings)) # If it is a character vector, it should be 'default'
  #   latent_dim <- 2^round(log(feature_dim / 4, 2)) # Divide feature_dim by 4 and round to the nearest power of 2
  # if (is.list(cvae_settings)) {
  #   if ("latent_dim" %in% names(cvae_settings))
  #     latent_dim <- cvae_settings$latent_dim
  #   else
  #     latent_dim <- 2^round(log(feature_dim / 4, 2)) # Divide feature_dim by 4 and round to the nearest power of 2
  # }
  #
  # cvae <- deepcombat_cvae(feature_dim, latent_dim,
  #                         n_hidden, n_batch, n_covariate)

  # # Creating optimizer
  # if (use_default_optim) {
  #   optimizer <- optim_adam(cvae$parameters, lr = 0.01, weight_decay = 0)
  # } else {
  #   optimizer <- NULL # If not using default, will have to pass to trainer manually
  # }
  #
  setup_object = list(dataloader = dl,
                      cvae = NULL,
                      optimizer = NULL,
                      input = input_list,
                      cvae_settings = setup_train$cvae_settings)
  class(setup_object) = "deepcombat_setup_object"
  return(setup_object)
}
