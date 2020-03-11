#' Create a model
#'
#' @param model The model to run ptmc.
#' @param data Data used in the calibration process
#' @param settings settings
#' @return Returns a list with the fist element being the mcmc samples formatting for analysis and plottig with the CODA package. The second is the log posterior value at each time step.
#'
#' @export
ptmc_func <- function(model, data, settings) {

  outPTpost <- vector(mode = "list", length = settings[["nrChains"]])
  outPTlp <- vector(mode = "list", length = settings[["nrChains"]])
  outPTtemp <- vector(mode = "list", length = settings[["nrChains"]])

  for (i in 1:settings[["nrChains"]]){
    out_raw <- run_ptmc(model, data, settings)
    out_post <- out_raw[,1:settings$P]
    colnames(out_post) <- model[["par_names"]]
    outPTpost[[i]] <- mcmc(out_post)
    
    outPTlp[[i]] <- out_raw[, settings$P+1]
    outPTtemp[[i]] <- out_raw[, settings$P+2]
  }
  
  outlpv <- data.frame(matrix(unlist(outPTlp), nrow=length(outPTlp[[1]])))
  colnames(outlpv) <- c(1:settings[["nrChains"]])
  outlpv <- outlpv %>% gather(colnames(outlpv), key="chain_no",value="lpost")
  outlpv$sample_no <-rep(1:length(outPTlp[[1]]), settings[["nrChains"]])

  outltempv <- data.frame(matrix(unlist(outPTtemp), nrow=length(outPTtemp[[1]])))
  colnames(outltempv) <- c(1:settings[["nrChains"]])
  outltempv <- outltempv %>% gather(colnames(outltempv), key="chain_no", value="temperature")
  outltempv$sample_no <- rep(1:length(outPTtemp[[1]]), settings[["nrChains"]])

  output <- list(
    mcmc=as.mcmc.list(outPTpost),
    lpost=outlpv,
    temp=outltempv
  )
  output
}
