# Comparing random sampling to set coverage

LiMMBo uses a subsampling step to estimate the overall genetic and non-genetic
covariance matrices, **Cg** and **Cn** for phenotypes **Y**. For phenotypes with *N* samples
and *P* traits, **Cg** and **Cn** are of size [*P* x *P*]. For the estimation of these
covariance matrices *P*(*P*+1)/2 parameters have to be estimated. Instead of
estimating these parameters from complete phenotype matrix **Y**, LiMMBo generates
[*s* x *s*] trait variance matrices by sampling *s* traits from the
total of *P* traits and applies the variance estimation to these small
matrices. It generates enough [*s* x *s*] sized matrices to ensure that each trait
combination is sampled at least a given number of times, eg 3.

In version 0.1.4, the subsampling is achieved by drawing random samples of size
*s* from *P*. For version 1.0.0, a subsamling method based on set coverag is used
to achieve coverage with near minimal number of subsampling matrices.

Evaluate_performance.py compares the two methods in terms of trait coverage,
the time for constructing the subsampling arrays (construction only; does not
include the variance estimation step) and the number of subsampling
matrices needed for coverage. The results are shown
[here](https://github.com/HannahVMeyer/limmbo/blob/master/performance/sampling.pdf).
