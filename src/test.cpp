
#include "Tree.h"
#include "Interval.h"

class Tree;
struct Interval;

arma::mat createDefaultPartition(arma::mat supp, u_int d, u_int k);

// [[Rcpp::export]]
arma::mat test(arma::mat data, int code) {

    Tree* tree = new Tree(data, code); // this will create the ENTIRE regression tree
    // tree->root will give the root node of the tree
    // double rootThresh = tree->root->getThreshold();
    // Rcpp::Rcout << "Tree SSE =  " << tree->getSSE() << std::endl;
    // Rcpp::Rcout << "Tree has " << tree->getLeaves() << " leaves" << std::endl;

    // tree->printTree();
    // Rcpp::Rcout<< "----------------------------------------" << std::endl;

    unsigned int nLeaves = tree->getLeaves();
    unsigned int d = tree->getNumFeats();
    unsigned int n = tree->getNumRows();  
    unsigned int k = 0;

    arma::uvec r = arma::conv_to<arma::uvec>::from(arma::linspace(0, n-1, n));
    arma::uvec c = arma::conv_to<arma::uvec>::from(arma::linspace(1, d, d));
    arma::mat  X = data.submat(r, c);
    arma::mat supp = support(X, d); // extract support
    arma::mat partition = createDefaultPartition(supp, d, nLeaves);

    // Rcpp::Rcout << "Created default partition: " <<
    //     partition.n_rows << " rows, " << partition.n_cols << " cols." << 
    //     std::endl;

    std::vector<Interval*> intervalStack; 
    std::unordered_map<u_int, arma::uvec> leafRowMap;
    dfs(tree->root, k, intervalStack, partition, supp, leafRowMap);
    /* after the above call to dfs(), the leafRowmap is fully populated and 
       can be used by the rest of the hybrid algorithm */

    // int n_points = 0;
    // for (u_int kk = 0; kk < k; kk++) {
    //     n_points += leafRowMap[kk].n_elem;
    //     Rcpp::Rcout << kk << "-th leaf has: " << leafRowMap[kk].n_elem << 
    //         "points" << std::endl;
    //     Rcpp::Rcout << leafRowMap[kk] << std::endl;
    // }
    // Rcpp::Rcout << n_points << " points total." << std::endl;
    // Rcpp::Rcout << "Filled " << k << " partition sets" << std::endl;

    delete(tree);
    return partition;
}

// [[Rcpp::export]]
arma::uvec sortOnFeatureSub(arma::mat data, arma::uvec rowvec, u_int d) {
    /* sortOnFeatureSub(): returns the row indices on the scale of the ORIGINAL
       row indices (1 to nrow(data)), rather than on the scale of subset. This
       allows us to avoid creating a new dataset as we go deeper into the tree,
       and instead just rely on keeping track of indices that give us sorted
       versions of the original feature when we subset (call .elem())

       data   : original full dataset; N x (D+1), where response is in 0-th col
       rowvec : vector of row indices that make up the subset that we consider
       d      : the index of the feature (column of the data) to sort on
    */
    arma::vec x = data.col(d);        // extract the d-th feature
    arma::vec xsub = x.elem(rowvec);  // extract the subset of the d-th feature
    // obtain the indices that place the d-th feature in ascending order
    arma::uvec sortedRowIndex = arma::sort_index(xsub);
    // Rcpp::Rcout << xsub.elem(sortedRowIndex) << std::endl;
    // return ORIGINAL indices arranged based on the order of the d-th feature
    return rowvec.elem(sortedRowIndex); 
}

// [[Rcpp::export]]
arma::uvec sortOnFeature(arma::mat data, u_int d) {
    // extract response from data --> data.col(0)
    // extract d-th feature from data --> data.col(d)

    arma::vec y = data.col(0);
    arma::vec x = data.col(d);
    u_int n = y.n_elem;
    arma::uvec rowlabel = arma::conv_to<arma::uvec>::from(
            arma::linspace(0, n-1, n));

    arma::uvec sortedRowIndex = arma::sort_index(x);
    return rowlabel.elem(sortedRowIndex);
}


// [[Rcpp::export]]
arma::mat sortDataOnFeature(arma::mat data, u_int d) {
    // do not subset data out because we later rely on (i think we rely on this?)
    // the d-th feature being in its corresponding column when we deal with splitting
    // we might able to get away with subsetting b/c i dont think we actually
    // make use of the left/right rows after splitting because they are only
    // "used" (but later not touched) in the initializations of the left/right nodes
    int n = data.n_rows;
    arma::vec rowid = arma::linspace(0, n-1, n);
    arma::mat out = arma::mat(data.n_rows, 3, arma::fill::zeros); 
    out.col(0) = data.col(0); // store data in first column
    out.col(1) = data.col(d); // store the d-th feature
    out.col(2) = rowid;

    arma::uvec rowidSorted = sortOnFeature(data, d);    


    arma::vec y = data.col(0);
    arma::vec x = data.col(d);
    
    out.col(0) = y.elem(rowidSorted);
    out.col(1) = x.elem(rowidSorted);
    out.col(2) = arma::conv_to<arma::vec>::from(rowidSorted);
    // out.col(2) = (data.col(2)).elem(rowidSorted);

    // split the data on the n-th row
    // print vector of indices that come before the n-th row
    // probably use linspace from element 0 to n; then subset the rowids
    int j = 5; // row of feature that we're going to split on (16th smallest value since vector sorted)
    int offset = 0; // this will take into account the min index that we split on to account for minBucket condition
    arma::uvec leftrows  = arma::conv_to<arma::uvec>::from(arma::linspace(0, j-1, j));
    arma::uvec rightrows = arma::conv_to<arma::uvec>::from(arma::linspace(j, n-1, n-j));

    Rcpp::Rcout<< "left: " << leftrows << std::endl;
    Rcpp::Rcout<< "right: " << rightrows << std::endl;

    /* these indices can be used to subset out the rows from the original data 
       and more importantly ** REPLACE THE INNER FOR LOOP ** that builds the
       left and right stacks */
    arma::uvec leftrowindex = rowidSorted.elem(leftrows);
    arma::uvec rightrowindex = rowidSorted.elem(rightrows);

    Rcpp::Rcout<< "left: " << leftrowindex << std::endl;
    Rcpp::Rcout<< "right: " << rightrowindex << std::endl;


    return out;
}


// [[Rcpp::export]]
arma::mat timeTree(arma::mat data) {

    Tree* tree = new Tree(data); // this will create the ENTIRE regression tree
    // tree->root will give the root node of the tree
    double rootThresh = tree->root->getThreshold();
    // Rcpp::Rcout << "Tree has " << tree->getLeaves() << " leaves" << std::endl;

    unsigned int nLeaves = tree->getLeaves();
    unsigned int d = tree->getNumFeats();
    unsigned int n = tree->getNumRows();  
    unsigned int k = 0;

    arma::uvec r = arma::conv_to<arma::uvec>::from(arma::linspace(0, n-1, n));
    arma::uvec c = arma::conv_to<arma::uvec>::from(arma::linspace(1, d, d));
    arma::mat  X = data.submat(r, c);
    arma::mat supp = support(X, d); // extract support
    arma::mat partition = createDefaultPartition(supp, d, nLeaves);

    std::vector<Interval*> intervalStack; 
    std::unordered_map<u_int, arma::uvec> leafRowMap;
    dfs(tree->root, k, intervalStack, partition, supp, leafRowMap);
    /* after the above call to dfs(), the leafRowmap is fully populated and 
       can be used by the rest of the hybrid algorithm */
    delete(tree);
    return partition;
}

 arma::mat createDefaultPartition(arma::mat supp, u_int d, u_int k) {
     /* 
            data : matrix w/ response in first column, [y | X]
            n :    # of data points
            d :    dimension of features
            k :    # of leaves
     */
    // arma::mat partition(k, 2 * d, arma::fill::zeros);
    arma::mat partition(2 * d, k, arma::fill::zeros);
    for (unsigned int i = 0; i < d; i++) {
        double lb = supp(i, 0);
        double ub = supp(i, 1);
        for (unsigned int r = 0; r < k; r++) {
            // partition(r, 2 * i) = lb;
            // partition(r, 2 * i + 1) = ub;
            partition(2 * i, r) = lb;
            partition(2 * i + 1, r) = ub;
        }
    }
    return partition;
 } // end createDefaultPartition() function


// compute the SSE associated with this decision node (rule)
double calculateRuleSSE(arma::mat z, arma::uvec leftRows, arma::uvec rightRows) {
    arma::vec y = z.col(0);
    // subset out y-values for left node
    arma::vec yLeft = y.elem(leftRows); 
    // subset out y-values for right node
    arma::vec yRight = y.elem(rightRows);
    // compute left-mean
    double leftMean = arma::mean(yLeft);
    // compute right-mean
    double rightMean = arma::mean(yRight);

    // Rcpp::Rcout<< "leftMean: " << leftMean << std::endl;
    // Rcpp::Rcout<< "rightMean: " << rightMean << std::endl;
    int nLeft = yLeft.n_elem;
    int nRight = yRight.n_elem;
    // compute sse for left
    double sseLeft = sse(yLeft, nLeft, leftMean);
    // compute sse for right
    double sseRight = sse(yRight, nRight, rightMean);
    // compute node sse
    // Rcpp::Rcout<< "sseLeft: " << sseLeft << std::endl;
    // Rcpp::Rcout<< "sseRight: " << sseRight << std::endl;
    return sseLeft + sseRight;
} // end calculateRuleSSE() function


// [[Rcpp::export]]
arma::uvec testBuildTree(arma::mat z, arma::uvec rows) {
    // note: this function is called every time a tree is built on 
    // a child node; each output will give the root node of the tree

    int nodeMinLimit = 0;
    // check terminating conditions to break out of recursive calls
    if (rows.n_elem <= nodeMinLimit) {
        // this->numLeaves++;
        Node* leaf = new Node(z, rows);
        // Rcpp::Rcout << "Leaf Node: " << leaf->getLeafVal() << std::endl;
        // return leaf; // return pointer to leaf Node
        return 0;
    }

    // else: not at leaf node, rather a decision  node
    /* iterate through features, rows (data) to find (1) optimal 
       splitting feat and (2) optimal splitting value for that feature
    */
    u_int numFeats = z.n_cols - 1;
    u_int numRows  = z.n_rows;
    double minSSE  = std::numeric_limits<double>::infinity();

    double optThreshold; // optimal split value
    u_int  optFeature;   // optimal feature to use in the split (col #)

    arma::uvec left, right;
    arma::uvec optLeft, optRight;
    for (u_int d = 1; d < numFeats; d++) { // loop thru features (cols)
        // start from col 1 b/c y in col 0

        for (u_int n = 0; n < numRows; n++) { // loop thru data (rows)
            // iterate through the rows corresponding to feature d 
            // to find the optimal value on which to partition (bisect) 
            // the data propose X(d, n) as the splitting rule

            double threshold = z(n, d); // propose new splitting value
            std::vector<u_int> rightRows, leftRows;

            // construct the left/right row index vectors 
            for (const auto &i : rows) {
                // compare val with each of the vals in other rows
                double val_i = z(i, d);
                if (val_i <= threshold) { 
                    // equality guarantees that at least 1 node will be 
                    // less than the threshold.. is this what we want? 
                    // if we want a case where we allow for 0 nodes in 
                    // threshold just have the inequality be strict
                    
                    // determine right child vs. left child membership 
                    // based on value compared to threshold
                    rightRows.push_back(i);
                } else {
                    leftRows.push_back(i);
                }

            } // end for() creating the left/right row index vectors

            if (rightRows.size() == 0 || leftRows.size() == 0) {
                // we do this check because we already checked the 
                // condition of a leaf node in beginning of function
                continue; // go back to start of INNER for over the rows
            } 

            // compute SSE associated with this decision rule
            // convert rightRow, leftRow into arma::uvec
            left  = arma::conv_to<arma::uvec>::from(leftRows);
            right = arma::conv_to<arma::uvec>::from(rightRows);

            double propSSE = calculateRuleSSE(z, left, right);
            // Rcpp::Rcout<< "left: " << left << std::endl;
            // Rcpp::Rcout<< "right: " << right << std::endl;
            // Rcpp::Rcout<< "sse: " << propSSE << std::endl;
            
            // TODO: isn't cp parameter in rpart measuring how much the 
            // improvement is? in that case, we should look at the diff
            // in the MSE rather than if we can find a smaller one
            if (propSSE < minSSE) {
                // Rcpp::Rcout<< "enter if" << std::endl;
                minSSE = propSSE;
                // TODO: store threshold value; this defines partition
                optThreshold = threshold;
                // TODO: store col number b/c this gives partition #
                // don't need to store the row (n) b/c we only need
                // the value (threshold) and the feature (d) so we know
                // where to define this 1-d interval in the partition
                optFeature = d;
                optLeft = left;
                optRight = right;
            } else {
                // ignore the rule
                // if we go the route of making a Rule object, 
                // we'll need to delete the rule here
            } // end if-else checking the new SSE
            
        } // end for() over data, over z(i, d), for i in [0, nRows]

    } // end for() over features ------ end OUTER FOR LOOP()

    // ************************************************************
    // IMPORTANT:
    // nalds code has part here which constructs the left/right rows, 
    // but i  think that;s because his rule doesnt store these; 
    // since we have these, we dont need to reconstruct them. 
    // just put the definitions outside of main loops
    // ************************************************************

    // DEBUG CODE TO LOOK AT VALUES:
    // Rcpp::Rcout<< "optimal feature: " << optFeature << std::endl;
    // Rcpp::Rcout<< "optimal split value: " << optThreshold << std::endl;
    // Rcpp::Rcout<< "min SSE: " << minSSE << std::endl;

    // construct node using optimal value, column, data, left, right
    Node* node = new Node(optThreshold, optFeature, z, optLeft, optRight);
    // build left and right nodes
    // in order to test one iteration of this function, just take out
    // left, right subtree function calls
    // node->left = buildTree(z, left);
    // node->right = buildTree(z, right);

    return optLeft;
} // end of buildTree() function


