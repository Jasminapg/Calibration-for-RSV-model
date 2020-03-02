// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

// run_ptmc
Eigen::MatrixXd run_ptmc(Rcpp::List model, Rcpp::RObject data, Rcpp::List settings);
RcppExport SEXP _ptmc_run_ptmc(SEXP modelSEXP, SEXP dataSEXP, SEXP settingsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type model(modelSEXP);
    Rcpp::traits::input_parameter< Rcpp::RObject >::type data(dataSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type settings(settingsSEXP);
    rcpp_result_gen = Rcpp::wrap(run_ptmc(model, data, settings));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_ptmc_run_ptmc", (DL_FUNC) &_ptmc_run_ptmc, 3},
    {NULL, NULL, 0}
};

RcppExport void R_init_ptmc(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}