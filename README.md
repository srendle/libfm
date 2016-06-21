libFM with Early Stopping
=========================

Changes:

* Removed mcmc and sgda optimization methods
* Added validation data as mandatory parameter
* Implemented early stopping of fm model
* Validation data is MANDATORY for this verions
* Removed handling of Meta 
* Removed regression task

Additional parameters:
* early stopping (bool) -- enabling early stopping
* num iterations for early stopping (int) -- how many iterations till break




libFM
=====

Library for factorization machines

web: http://www.libfm.org/

forum: https://groups.google.com/forum/#!forum/libfm

Factorization machines (FM) are a generic approach that allows to mimic most factorization models by feature engineering. This way, factorization machines combine the generality of feature engineering with the superiority of factorization models in estimating interactions between categorical variables of large domain. libFM is a software implementation for factorization machines that features stochastic gradient descent (SGD) and alternating least squares (ALS) optimization as well as Bayesian inference using Markov Chain Monte Carlo (MCMC).

Compile
=======
libFM has been tested with the GNU compiler collection and GNU make. libFM and the tools can be compiled with
> make all

Usage
=====
Please see the [libFM 1.4.2 manual](http://www.libfm.org/libfm-1.42.manual.pdf) for details about how to use libFM. If you have questions, please visit the [forum](https://groups.google.com/forum/#!forum/libfm).
