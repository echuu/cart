// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include "cart_types.h"
#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// old_build
void old_build(arma::mat data);
RcppExport SEXP _cart_old_build(SEXP dataSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type data(dataSEXP);
    old_build(data);
    return R_NilValue;
END_RCPP
}
// old_partition
arma::mat old_partition(arma::mat data);
RcppExport SEXP _cart_old_partition(SEXP dataSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type data(dataSEXP);
    rcpp_result_gen = Rcpp::wrap(old_partition(data));
    return rcpp_result_gen;
END_RCPP
}
// f_build
void f_build(arma::mat data, bool code);
RcppExport SEXP _cart_f_build(SEXP dataSEXP, SEXP codeSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type data(dataSEXP);
    Rcpp::traits::input_parameter< bool >::type code(codeSEXP);
    f_build(data, code);
    return R_NilValue;
END_RCPP
}
// f_partition
arma::mat f_partition(arma::mat data, bool code);
RcppExport SEXP _cart_f_partition(SEXP dataSEXP, SEXP codeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type data(dataSEXP);
    Rcpp::traits::input_parameter< bool >::type code(codeSEXP);
    rcpp_result_gen = Rcpp::wrap(f_partition(data, code));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_cart_old_build", (DL_FUNC) &_cart_old_build, 1},
    {"_cart_old_partition", (DL_FUNC) &_cart_old_partition, 1},
    {"_cart_f_build", (DL_FUNC) &_cart_f_build, 2},
    {"_cart_f_partition", (DL_FUNC) &_cart_f_partition, 2},
    {NULL, NULL, 0}
};

RcppExport void R_init_cart(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
