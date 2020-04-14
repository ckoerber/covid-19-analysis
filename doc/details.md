# Details

Additionally it allows to crosscheck the effect of fitting new admissions.

## Data

Notebooks in this repo make use of the  [NYC Department of Health and Mental Hygiene (DOHMH) repo data](https://github.com/nychealth/coronavirus-data).
In particular the `case-hosp-death.csv` is used.


## Models

This repository implements the regular SIR model and 3 variations.
* Standard SIR
* SIR model with time-dependent `beta` parameter (`beta(t) = beta(0) * f(t)`, where `f(t)` is a [logistic function](https://en.wikipedia.org/wiki/Logistic_function))
* "SIHR" model which is explained below
* "SIHR" model with time-dependent `beta` parameter

### The SIHR model

The SIHR model is a variation of the SIR model which explicitly encodes the number of hospitalizations `H` as a new compartment.
```
    S -> I ------> R
         I -> H -> R
```
This allows to analyze the effect of
* temporal delays between being infected and being admitted to a hospital and
* hospital capacity limits

SIHR introduces two new parameters compared to the regular SIR model,
the rate of hospitalizations `beta_h` and the recovery rate when hospitalized `gamma_H`.
It is the assumption that the rate of hospitalizations `beta_h` is proportional to the number of infections and similar to the SIR model, the recovery rate is proportional to the number of hospitalizations.
In particular, the set of model equations is
```
S(t + dt) = S(t) - beta_I S(t) I(t)
I(t + dt) = I(t) + beta_I S(t) I(t) - beta_h I(t) - gamma_I I(t)
H(t + dt) = H(t) + beta_h I(t) - gamma_H H(t)
R(t + dt) = R(t) + gamma_I I(t) + gamma_H H(t)
```
Furthermore, the hospitalization rate `beta_h` is affected by the hospital capacity `C` such that `beta_h = 0` if `H(t) > C`.


## Room for improvements

### Parameter and data distributions are approximated by uncorrelated normal distributions

While in the limit of large numbers this approximation will improve, it is known that a few parameter distributions do not follow normal distribution.
Propagating experimental parameter distributions might increase accuracy; likely, this would require a MCMC implementation of the fit.

### Data uncertainty

While I believe that fitting daily new admissions are likely the most accurate source of information (polling numbers indicate that in some countries, the number of known cases is underestimated by a factor of more than two), I was just able to use a conservative estimate for temporal correlations in these numbers.
A better estimation of such uncertainties would strengthen the extrapolation.
