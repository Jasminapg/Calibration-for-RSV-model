#' Create a model
#'
#' @param model The model to run ptmc for (build using \code{\link{build_model}}).
#' @param data Data used in the calibration process
#' @param settings settings
#' @return Returns a model with the approximate log likelihood value calculated using SMC (`model$logLikelihood`). If `history` is set to `True` then the history is stored in `model$particle`s.
#'
#' @export
ptmc_func <- function(model, data, settings) {

  outPTpost <- vector(mode = "list", length = settings[["nrChains"]])
  outPTlp <- vector(mode = "list", length = settings[["nrChains"]])

  for (i in 1:settings[["nrChains"]]){
    out_raw <- run_ptmc(model, data, settings)
    out_post <- out_raw[,1:settings$P]
    outPTlp[[i]] <- out_raw[, settings$P+1]
    colnames(out_post) <- model[["par_names"]]
    outPTpost[[i]] <- mcmc(out_post)
  }
  
  outlpv <- data.frame(matrix(unlist(outPTlp), nrow=length(outPTlp[[1]])))
  colnames(outlpv) <- c(1:settings[["nrChains"]])
  outlpv <- outlpv %>% gather(colnames(outlpv), key="chain_no",value="lpost")
  outlpv$sample_no <-rep(1:length(outPTlp[[1]]), settings[["nrChains"]])

  output <- list(
    mcmc=as.mcmc.list(outPTpost),
    lpost=outlpv
  )
  output
}
