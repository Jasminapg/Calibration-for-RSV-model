#include <Rcpp.h>
#include <RcppEigen.h>

#include "ptmc.hpp"
// [[Rcpp::depends(RcppEigen)]]

using RPTMC = ptmc::PTMC<Rcpp::RObject>;

void init_gen_init(RPTMC* model, Rcpp::Function gen_init) {
  auto func = [gen_init]() {
    PutRNGstate();
    auto rData = gen_init();
    GetRNGstate();
    return Rcpp::as<VectorXd>(rData);
  };
  model->gen_init = func;
}

void init_eval_lpr(RPTMC* model, Rcpp::Function eval_lpr) {
  auto func = [eval_lpr](VectorXd params) {
    PutRNGstate();
    auto rData = eval_lpr(params);
    GetRNGstate();
    return Rcpp::as<double>(rData);
  };
  model->eval_lpr = func;
}

void init_eval_ll(RPTMC* model, Rcpp::Function eval_ll) {
  auto func = [eval_ll](Rcpp::RObject data, VectorXd params) {
    PutRNGstate();
    auto rData = eval_ll(data, params);
    GetRNGstate();
    return Rcpp::as<double>(rData);
  };
  model->eval_ll = func;
}


// [[Rcpp::export]]
Eigen::MatrixXd run_ptmc(Rcpp::List model, Rcpp::RObject data, Rcpp::List settings)
  {

  RPTMC PTMC;
  init_gen_init(&PTMC, model[0]);
  init_eval_lpr(&PTMC, model[1]);
  init_eval_ll(&PTMC, model[2]);

  MatrixXd output;
  output = PTMC.run_ptmc_C(PTMC, data, settings);

  return output;
}
