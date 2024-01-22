deepcombat_encoder <- nn_module(
  "DeepComBat Encoder",
  initialize = function(vae_dim, n_batch, n_covariate) {
    self$n_layers <- length(vae_dim)
    self$layers <- nn_module_list()
    self$layers[[1]] <- nn_linear(vae_dim[1] + n_batch + n_covariate, vae_dim[2])

    if (self$n_layers > 3) {
      for (i in 2:(self$n_layers - 2)) {
        self$layers[[i]] <- nn_linear(vae_dim[i], vae_dim[i + 1])
      }
    }

    self$e_mu <- nn_linear(vae_dim[self$n_layers - 1], vae_dim[self$n_layers])
    self$e_logvar <- nn_linear(vae_dim[self$n_layers - 1], vae_dim[self$n_layers])
  },

  forward = function(features, batch = NULL, covariates = NULL) {
    tmp <- torch_cat(list(features, batch, covariates), dim = -1)
    for (i in 1:(self$n_layers - 2)) {
      tmp <- torch_tanh(self$layers[[i]](tmp))
    }

    # encode hidden layer to mean and variance vectors
    mu <- self$e_mu(tmp)
    logvar <- self$e_logvar(tmp)
    return(list(mu, logvar))
  })

deepcombat_decoder <- nn_module(
  "DeepComBat Decoder",
  initialize = function(vae_dim, n_batch, n_covariate) {
    self$n_layers = length(vae_dim)
    self$layers <- nn_module_list()

    self$layers[[1]] <- nn_linear(vae_dim[self$n_layers] + n_batch + n_covariate,
                                  vae_dim[self$n_layers - 1])

    for (i in 2:(self$n_layers - 1)) {
      self$layers[[i]] <- nn_linear(vae_dim[self$n_layers + 1 - i],
                                    vae_dim[self$n_layers - i])
    }
  },
  forward = function(z, batch = NULL, covariates = NULL) {
    tmp <- torch_cat(list(z, batch, covariates), dim = -1)

    for (i in 1:(self$n_layers - 2)) {
      tmp <- torch_tanh(self$layers[[i]](tmp))
    }

    output <- self$layers[[self$n_layers - 1]](tmp)

    return(output)
  })

#' DeepComBat CVAE architecture
#'
#' Code for CVAE architecture. This function is called through helper functions of deepcombat_setup, deepcombat_trainer and deepcombat_harmonize.
#'
#' @param feature_dim Number of features to harmonize
#' @param latent_dim Desired latent space size
#' @param n_hidden Number of hidden layers
#' @param n_batch Number of batches
#' @param n_covariate Number of covariates
#' @param vae_dim Custom VAE layer sizes of length (n_hidden + 2). Layers must be strictly decreasing in size.
#'
#' @return nn_module object with desired architecture.
#' @export
#'
#' @import torch
#' @import neuroCombat
deepcombat_cvae <- nn_module(
  "DeepComBat CVAE",
  initialize = function(feature_dim, latent_dim,
                        n_hidden, n_batch, n_covariate,
                        vae_dim = NULL) {
    self$latent_dim <- latent_dim
    self$n_batch <- n_batch
    self$n_covariate <- n_covariate
    if (is.null(vae_dim)) {
      self$dims <- self$calculate_vae_dim(feature_dim, latent_dim, n_hidden)
    } else {
      if ((length(vae_dim) != n_hidden + 2)) {
        stop("vae_dim must be of length n_hidden + 2 with first element being input size and second element being latent size.")
      }
      if (any(diff(vae_dim) >= 0)) {
        stop("vae_dim must be strictly decreasing.")
      }
      if (vae_dim[1] != feature_dim) {
        stop("First element of vae_dim must be equal to input size.")
      }
      self$dims <- vae_dim
    }

    self$encoder <- deepcombat_encoder(self$dims, n_batch, n_covariate)
    self$decoder <- deepcombat_decoder(self$dims, n_batch, n_covariate)
  },
  forward = function(item_list) {
    features <- item_list[[1]]
    batch <- item_list[[3]]
    covariates <- NULL
    if (self$n_covariate != 0) {
      covariates <- item_list[[2]]
    }

    # encode features to latent feature distributions
    latent_dist <- self$encoder(features = features, batch = batch,
                                covariates = covariates)
    feat_mu <- latent_dist[[1]]
    feat_logvar <- latent_dist[[2]]

    # sample from latent distribution with re-parameterization trick
    feat_z <- feat_mu + torch_exp(feat_logvar * 0.5) * torch_randn_like(feat_mu)

    feat_reconstructed <- self$decoder(z = feat_z, batch = batch,
                                       covariates = covariates)

    return(list(feat_recon = feat_reconstructed,
                feat_mu = feat_mu,
                feat_logvar = feat_logvar,
                feat_z = feat_z))
  },
  calculate_vae_dim = function(input_dim, latent_dim, n_hidden) {
    if (input_dim <= latent_dim) {
      stop("Latent dimension must be smaller than input dimension")
    }
    if (n_hidden == 0) {
      return(c(input_dim, latent_dim))
    }
    hidden_dim <- rep(0, n_hidden)
    range <- input_dim - latent_dim
    for (i in 1:n_hidden) {
      hidden_dim[i] <- latent_dim + floor(range * (n_hidden - i + 1) / (n_hidden + 1))
    }

    return(c(input_dim, hidden_dim, latent_dim))
  },
  harmonize = function(setup_obj,
                       correct = c("combat", "combat"),
                       verbose = FALSE) {
    ## Check parameters
    correct[1] <- match.arg(correct[1], c("combat", "covbat"))
    correct[2] <- match.arg(correct[2], c("combat", "covbat"))

    ## Setup
    torch_ds <- setup_obj$dataloader$dataset
    raw_means <- torch_tensor(attr(setup_obj$input$features,
                                   which = "scaled:center"))
    raw_sds <- torch_tensor(attr(setup_obj$input$features,
                                 which = "scaled:scale"))
    batch <- torch_ds$batch
    target_batch <- torch_ds$target_batch
    covariates <- NULL
    if (self$n_covariate != 0) {
      covariates <- torch_ds$covariates
    }

    combat_batch_list <- self$convert_batch_for_combat(batch, target_batch)

    # Set reference batch so ComBat can interpret it
    if (as.matrix(target_batch)[1] %% 1 != 0) { # Checks if desired reference_batch is NULL (If NULL, target_batch will have non-integers)
      reference_batch <- NULL
    } else {
      reference_batch <- combat_batch_list$target_batch[1]
    }

    ## Get latent space and correct
    latent_dist <- self$encoder(features = torch_ds$features, batch = batch,
                                covariates = covariates)
    feat_mu <- latent_dist[[1]]
    feat_logvar <- latent_dist[[2]]

    if (verbose) {
      cat("Harmonizing latent space\n")
    }
    if (correct[1] == "combat") {
      latent_combat <- neuroCombat(dat = t(as.matrix(feat_mu)),
                                   batch = combat_batch_list$batch,
                                   mod = as.matrix(torch_ds$covariates),
                                   ref.batch = reference_batch,
                                   verbose = verbose)
      latent_estimates <- latent_combat$estimates
      corrected_mu <- torch_tensor(t(latent_combat$dat.combat))
    } else if (correct[1] == "covbat") {
      cat("Harmonizing latent space with CovBat\n")
      if (!is.null(reference_batch)) {
        stop("Cannot yet run CovBat with reference batch setting.")
      }
      corrected_mu <- torch_tensor(t(covbat(dat = t(as.matrix(feat_mu)),
                                            bat = combat_batch_list$batch,
                                            mod = as.matrix(torch_ds$covariates),
                                            verbose = verbose)$dat.covbat))
    }

    ## Decode latent space
    if (verbose) {
      cat("Decoding harmonized latent space\n")
    }
    feat_reconstructed <- self$decoder(z = feat_mu, batch = batch,
                                       covariates = covariates) * raw_sds + raw_means
    feat_restyled <- self$decoder(z = feat_mu, batch = target_batch,
                                  covariates = covariates) * raw_sds + raw_means
    feat_combat_restyled <- self$decoder(z = corrected_mu, batch = target_batch,
                                         covariates = covariates) * raw_sds + raw_means

    ## Correct residuals from autoencoder
    feat_resids <- torch_ds$features * raw_sds + raw_means - feat_reconstructed
    if (correct[2] == "combat") {
      if (verbose) {
        cat("Harmonizing autoencoder residuals using ComBat\n")
      }
      residual_combat <- neuroCombat(dat = t(as.matrix(feat_resids)),
                                     batch = combat_batch_list$batch,
                                     mod = as.matrix(torch_ds$covariates),
                                     ref.batch = reference_batch,
                                     verbose = verbose)
      residual_estimates <- residual_combat$estimates
      corrected_resids <- torch_tensor(t(residual_combat$dat.combat))
    }
    if (correct[2] == "covbat") {
      if (verbose) {
        cat("Harmonizing autoencoder residuals using CovBat\n")
      }
      corrected_resids <- torch_tensor(t(covbat(dat = t(as.matrix(feat_resids)),
                                                bat = combat_batch_list$batch,
                                                mod = as.matrix(torch_ds$covariates),
                                                verbose = verbose)$dat.covbat))
    }

    harmonize_object <- list(harmonized = as.matrix(feat_combat_restyled + corrected_resids),
                             latent_logvar = as.matrix(feat_logvar),
                             additional_outputs = list(raw_means = raw_means,
                                                       raw_sds = raw_sds,
                                                       latent_estimates = latent_estimates,
                                                       residual_estimates = residual_estimates,
                                                       unharmonized_latent_representation = as.matrix(feat_mu),
                                                       unharmonized_reconstructions = as.matrix(feat_reconstructed),
                                                       unharmonized_residuals = as.matrix(feat_resids),
                                                       harmonized_latent_representation = as.matrix(corrected_mu),
                                                       harmonized_reconstructions = as.matrix(feat_combat_restyled),
                                                       harmonized_resids = as.matrix(corrected_resids)))
    class(harmonize_object) = "deepcombat_harmonize_object"
    return(harmonize_object)
  },
  harmonize_from_train = function(setup_obj,
                                  harmonize_train_obj,
                                  correct = c("combat", "combat"),
                                  verbose = FALSE) {
    ## Check parameters
    correct[1] <- match.arg(correct[1], c("combat", "covbat"))
    correct[2] <- match.arg(correct[2], c("combat", "covbat"))

    ## Setup
    torch_ds <- setup_obj$dataloader$dataset
    raw_means <- torch_tensor(attr(setup_obj$input$features,
                                   which = "scaled:center"))
    raw_sds <- torch_tensor(attr(setup_obj$input$features,
                                 which = "scaled:scale"))
    batch <- torch_ds$batch
    target_batch <- torch_ds$target_batch
    covariates <- NULL
    if (self$n_covariate != 0) {
      covariates <- torch_ds$covariates
    }

    combat_batch_list <- self$convert_batch_for_combat(batch, target_batch)

    # Set reference batch so ComBat can interpret it
    if (as.matrix(target_batch)[1] %% 1 != 0) { # Checks if desired reference_batch is NULL (If NULL, target_batch will have non-integers)
      reference_batch <- NULL
    } else {
      reference_batch <- combat_batch_list$target_batch[1]
    }

    ## Get latent space and correct
    latent_dist <- self$encoder(features = torch_ds$features, batch = batch,
                                covariates = covariates)
    feat_mu <- latent_dist[[1]]
    feat_logvar <- latent_dist[[2]]

    if (verbose) {
      cat("Harmonizing latent space\n")
    }
    if (correct[1] == "combat") {
      latent_combat <- self$combat_from_train(dat = t(as.matrix(feat_mu)),
                                              batch = combat_batch_list$batch,
                                              mod = as.matrix(torch_ds$covariates),
                                              estimates = harmonize_train_obj$additional_outputs$latent_estimates,
                                              verbose = verbose)
      latent_estimates <- harmonize_train_obj$additional_outputs$latent_estimates
      corrected_mu <- torch_tensor(t(latent_combat$dat.combat))
    } else if (correct[1] == "covbat") {
      stop("Cannot yet run CovBat on validation data.")
      cat("Harmonizing latent space with CovBat\n")
      if (!is.null(reference_batch)) {
        stop("Cannot yet run CovBat with reference batch setting.")
      }
      corrected_mu <- torch_tensor(t(covbat(dat = t(as.matrix(feat_mu)),
                                            bat = combat_batch_list$batch,
                                            mod = as.matrix(torch_ds$covariates),
                                            verbose = verbose)$dat.covbat))
    }

    ## Decode latent space
    if (verbose) {
      cat("Decoding harmonized latent space\n")
    }
    feat_reconstructed <- self$decoder(z = feat_mu, batch = batch,
                                       covariates = covariates) * raw_sds + raw_means
    feat_restyled <- self$decoder(z = feat_mu, batch = target_batch,
                                  covariates = covariates) * raw_sds + raw_means
    feat_combat_restyled <- self$decoder(z = corrected_mu, batch = target_batch,
                                         covariates = covariates) * raw_sds + raw_means

    ## Correct residuals from autoencoder
    feat_resids <- torch_ds$features * raw_sds + raw_means - feat_reconstructed
    if (correct[2] == "combat") {
      if (verbose) {
        cat("Harmonizing autoencoder residuals using ComBat\n")
      }
      residual_combat <- self$combat_from_train(dat = t(as.matrix(feat_resids)),
                                                batch = combat_batch_list$batch,
                                                mod = as.matrix(torch_ds$covariates),
                                                estimates = harmonize_train_obj$additional_outputs$residual_estimates,
                                                verbose = verbose)
      residual_estimates <- harmonize_train_obj$additional_outputs$residual_estimates
      corrected_resids <- torch_tensor(t(residual_combat$dat.combat))
    }
    if (correct[2] == "covbat") {
      stop("Cannot yet run CovBat with validation data.")
      if (verbose) {
        cat("Harmonizing autoencoder residuals using CovBat\n")
      }
      corrected_resids <- torch_tensor(t(covbat(dat = t(as.matrix(feat_resids)),
                                                bat = combat_batch_list$batch,
                                                mod = as.matrix(torch_ds$covariates),
                                                verbose = verbose)$dat.covbat))
    }

    harmonize_object <- list(harmonized = as.matrix(feat_combat_restyled + corrected_resids),
                             latent_logvar = as.matrix(feat_logvar),
                             additional_outputs = list(raw_means = raw_means,
                                                       raw_sds = raw_sds,
                                                       latent_estimates = latent_estimates,
                                                       residual_estimates = residual_estimates,
                                                       unharmonized_latent_representation = as.matrix(feat_mu),
                                                       unharmonized_reconstructions = as.matrix(feat_reconstructed),
                                                       unharmonized_residuals = as.matrix(feat_resids),
                                                       harmonized_latent_representation = as.matrix(corrected_mu),
                                                       harmonized_reconstructions = as.matrix(feat_combat_restyled),
                                                       harmonized_resids = as.matrix(corrected_resids)))
    class(harmonize_object) = "deepcombat_harmonize_object"
    return(harmonize_object)
  },
  get_vae_transforms = function(output) {
    list(recon = output$feat_recon,
         mu_2 = output$feat_mu$pow(2),
         logvar = output$feat_logvar,
         var = output$feat_logvar$exp(),
         n_minibatch = dim(output$feat_recon)[1])
  },
  get_recon_loss = function(transform_list, target) {
    nn_mse_loss(reduction = "sum")(transform_list$recon, target)
  },
  get_prior_loss = function(transform_list) {
    return(0.5 * torch_sum((transform_list$mu_2 + transform_list$var - 1 - transform_list$logvar)))
  },
  combat_from_train = function (dat, batch, mod = NULL, estimates = NULL,
                                verbose = FALSE) {
    dat <- as.matrix(dat)
    batch <- as.factor(batch)
    n.array <- length(batch)
    n.batch <- nlevels(batch)
    batches <- lapply(levels(batch), function(x) which(batch == x))
    n.batches <- sapply(batches, length)

    design <- cbind(model.matrix(~-1 + batch), mod)

    batch_train <- as.factor(estimates$batch)
    batches_train <- lapply(levels(batch_train), function(x) which(batch_train == x))
    n.proportions_train <- sapply(batches_train, length) / length(batch_train)

    B.hat <- estimates$beta.hat
    var.pooled <- estimates$var.pooled

    if (!is.null(design)) {
      tmp <- design
      if (is.null(estimates$ref.batch)) {
        tmp[, c(1:n.batch)] <- matrix(n.proportions_train,
                                      nrow = nrow(design),
                                      ncol = n.batch, byrow = T)
      } else {
        tmp[, c(1:n.batch)] <- 0
        tmp[, which(levels(batch) == estimates$ref.batch)] <- 1
      }
      stand.mean <- t(tmp %*% B.hat)
    }
    s.data <- (dat - stand.mean)/(tcrossprod(sqrt(var.pooled),
                                             rep(1, n.array)))
    batch.design <- design[, 1:n.batch]
    gamma.star <- estimates$gamma.star
    delta.star <- estimates$delta.star

    bayesdata <- s.data
    j <- 1
    for (i in batches) {
      bayesdata[, i] <- (bayesdata[, i] - t(batch.design[i, ] %*% gamma.star)) /
        tcrossprod(sqrt(delta.star[j, ]), rep(1, n.batches[j]))
      j <- j + 1
    }
    bayesdata <- (bayesdata * (tcrossprod(sqrt(var.pooled), rep(1, n.array)))) + stand.mean

    return(list(dat.combat = bayesdata, stand.mean = stand.mean, estimates = estimates))
  },
  convert_batch_for_combat = function(batch, target_batch) { # Convert one-hot encoded batch to label encoded batch
    indicator <- matrix(rep(1:(dim(batch)[2]), dim(batch)[1]),
                        ncol = dim(batch)[2], byrow = TRUE)

    combat_batch <- rowSums(as.matrix(batch) * indicator)
    combat_target_batch <- rowSums(as.matrix(target_batch) * indicator)

    return(list(batch = combat_batch,
                target_batch = combat_target_batch))
  })
