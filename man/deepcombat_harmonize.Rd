% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/deepcombat_trainer.R
\name{deepcombat_harmonize}
\alias{deepcombat_harmonize}
\title{DeepComBat Harmonization}
\usage{
deepcombat_harmonize(
  setup_obj,
  trained_model,
  harmonize_obj = NULL,
  plot_logvars = FALSE,
  verbose = FALSE
)
}
\arguments{
\item{setup_obj}{A deepcombat_setup_object.}

\item{trained_model}{A trained DeepComBat CVAE model from deepcombat_trainer.}

\item{harmonize_obj}{Optional output from deepcombat_harmonize run on the training data. This output contains the pre-trained DeepComBat model as well as latent-space and residual ComBat estimands from the training data.}

\item{plot_logvars}{A logical value indicating whether to plot the density of logvars according to dimension -- this may be useful for manual hyperparameter tuning. Default is FALSE.}

\item{verbose}{A logical value indicating whether to print progress messages. Default is FALSE.}
}
\value{
A deepcombat_harmonized_object containing the harmonized data, logvars, and additional outputs that may be useful for methodologists/advanced users.
}
\description{
This function performs harmonization using DeepComBat. It takes in a setup object and a trained DeepComBat CVAE model, and returns the harmonized output.
}
\examples{
\dontrun{
harmonized_output <- deepcombat_harmonize(setup_obj, trained_model,
plot_logvars = TRUE, verbose = TRUE)
}

}
