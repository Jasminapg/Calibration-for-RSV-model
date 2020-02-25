#' Create a model
#'
#' @param model The model to run ptmc for (build using \code{\link{build_model}}).
#' @param data Data used in the calibration process
#' @param settings settings
#' @return Returns a model with the approximate log likelihood value calculated using SMC (`model$logLikelihood`). If `history` is set to `True` then the history is stored in `model$particle`s.
#'
#' @export
ptmc_func <- function(model, data, settings) {

  output <- run_ptmc(model, data, settings)
  output # posterior samples, loglikelihood
}
