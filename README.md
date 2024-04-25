# Inputs Dimension Reduction for Predictive Modeling 
We consider using PCA, PLS, AS, gKDR, gKDR-i, and gKDR-v.


# Description

In this book chapter, the focus is to employ the methods (PCA, PLS, AS, gKDR, gKDR-i, and gKDR-v) to find a new set of inputs to fit a predictive model, Gaussian process regression. We employed two phases to the analysis: the dimension reduction stage of each method and the Gaussian Process Regression(GPR) fit based on the reduced inputs.


## First Stage

 We used the said methods to estimate the projection matrix $W_D$ to obtain a new set of $D-$ predictors, $Z = X W_D$ to fit a predictive model. Since each method has its way of constructing the low-dimensional approximations to the simulators, we will proceed to find the Frobenius norm to evaluate the discrepancy between the estimated projection matrix  $W_d$ and the true projector $W_0$ which is the key idea of the study.


## Second Stage

For the second phase, we used the reduced inputs to fit our GPR. The same training set is utilized to fit the model and employ the test set to make predictions. We measured the predictive performance using root mean square error prediction (RMSEP).


# Codes Usage
 
 ## Train_data

 This folder contains the train set to do the dimension reduction

 ## gKDR-Codes

 This folder contains the MATLAB code to do the dimension reduction of the gKDR and the two variants 


 ## Numerical Illustration Codes

 This is Rmarkdown file that contains data generations, dimension reduction based on PCA, PLS and AS. It also contains the code that fit the GPR based on the reduced inputs.

# Reference

K. Fukumizu and C.Leng (2013) Gradient-based kernel dimension reduction for regression. Journal of the American Statistical Association, to appear. 
