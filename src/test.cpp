
#include "Tree.h"
#include "Interval.h"
#include <omp.h>

// [[Rcpp::plugins(openmp)]]



// [[Rcpp::export]]
void old_build(arma::mat data) {

    Tree* tree = new Tree(data);
    // std::unordered_map<u_int, arma::vec>* pmap = tree->getPartition();
	// std::unordered_map<u_int, arma::uvec>* leafRowMap = tree->getLeafRowMap();
	// unsigned int nLeaves = tree->getLeaves();
    // unsigned int d = tree->getNumFeats();

    // // turn map into array with partitions populating each row
    // arma::mat partmatrix(2 * d, nLeaves);
    // for (u_int i = 0; i < nLeaves; i++) {
    //     partmatrix.col(i) = (*pmap)[i];
    // }

    // tree->printTree();

    // return partmatrix;
}


// [[Rcpp::export]]
arma::mat old_partition(arma::mat data) {

    Tree* tree = new Tree(data);
    std::unordered_map<u_int, arma::vec>* pmap = tree->getPartition();
	std::unordered_map<u_int, arma::uvec>* leafRowMap = tree->getLeafRowMap();
	unsigned int nLeaves = tree->getLeaves();
    unsigned int d = tree->getNumFeats();

    // turn map into array with partitions populating each row
    arma::mat partmatrix(2 * d, nLeaves);
    for (u_int i = 0; i < nLeaves; i++) {
        partmatrix.col(i) = (*pmap)[i];
    }

    tree->printTree();

    return partmatrix;
}

// [[Rcpp::export]]
void f_build(arma::mat data, bool code) {

    Tree* tree = new Tree(data, code);

    // std::unordered_map<u_int, arma::vec>* pmap = tree->getPartition();
	// std::unordered_map<u_int, arma::uvec>* leafRowMap = tree->getLeafRowMap();
	// unsigned int nLeaves = tree->getLeaves();
    // unsigned int d = tree->getNumFeats();

    // // turn map into array with partitions populating each row
    // arma::mat partmatrix(2 * d, nLeaves);
    // for (u_int i = 0; i < nLeaves; i++) {
    //     partmatrix.col(i) = (*pmap)[i];
    // }

    // tree->printTree();

    // return partmatrix;
}


// [[Rcpp::export]]
arma::mat f_partition(arma::mat data, bool code) {

    Tree* tree = new Tree(data, code);

    std::unordered_map<u_int, arma::vec>* pmap = tree->getPartition();
	std::unordered_map<u_int, arma::uvec>* leafRowMap = tree->getLeafRowMap();
	unsigned int nLeaves = tree->getLeaves();
    unsigned int d = tree->getNumFeats();

    // turn map into array with partitions populating each row
    arma::mat partmatrix(2 * d, nLeaves);
    for (u_int i = 0; i < nLeaves; i++) {
        partmatrix.col(i) = (*pmap)[i];
    }

    tree->printTree();

    return partmatrix;

}

