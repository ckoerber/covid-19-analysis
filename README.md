# COVID-19 Analysis of NYC Data

This repo aims to provide tools which allow to characterize uncertainties in estimations of COVID-19 hospital admissions based on the [CHIME model](https://github.com/CodeForPhilly/chime).

Implemented tools utilize the [SIR model](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIR_model) and variations which add effects of social distancing measures and delay in hospitalizations.
Simulations of these models are fitted to data in a statistical context which allows to extract model parameter distributions for, e.g., social distancing policies and initial infections.
Fitted models are applied to data provided by the [NYC Department of Health and Mental Hygiene (DOHMH)](https://github.com/nychealth/coronavirus-data) to predict a 1-2 week window.


## Disclaimer

I do not have a background in epidemiology or related fields.
For now, the analysis was **not cross-checked and reviewed by an expert in the field**.
While I am confident that predictions are stable in a time-frame for 1-2 weeks from a data-driven point of view, I might have missed important details specific to this problem.
If you are familiar with this field, please reach out and provide feedback.

## Conclusion

1. Fitting daily new admissions of the NYC data makes it possible to consistently predict admissions in a 1-2 week window.
If social measures do not change, the prediction window might be extended.
2. Social distancing measures are essential to describe the data. To reliably fit social distancing model parameters, one has to "see a bend" in admissions.
3. Delay in hospitalizations (SIR vs. SIHR)...


## Details

Additionally it allows to crosscheck the effect of fitting new admissions.

### Data

Notebooks in this repo make use of the  [NYC Department of Health and Mental Hygiene (DOHMH) repo data](https://github.com/nychealth/coronavirus-data).
In particular the `case-hosp-death.csv` is used.


### Models

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
This allows to analyze the effect of temporal delays between being infected and being admitted to a hospital and hospital capacity limits

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

#### Parameter and data distributions are approximated by uncorrelated normal distributions

While in the limit of large numbers this approximation will improve, it is known that a few parameter distributions do not follow normal distribution.
Propagating experimental parameter distributions might increase accuracy; likely, this would require a MCMC implementation of the fit.

#### Data uncertainty

While I believe that fitting daily new admissions are likely the most accurate source of information (polling numbers indicate that in some countries, the number of known cases is underestimated by a factor of more than two), I was just able to use a conservative estimate for temporal correlations in these numbers.
A better estimation of such uncertainties would strengthen the extrapolation.

## Repository content

### Computation

Folder | Description
---|---
`models` | Implementation of the SIR and SIHR model including wrappers which simplify fitting
`utils` | Utility functions for loading data and plotting fits
`tests` | Unit tests for implemented models

### Analysis notebooks

File | Description | Conclusion
---|---|---
`SIR-penn-chime-benchmark` | Comparison of SIR model in this repository against `penn_chime` | Both modules agree with and without social distancing measures.
`SIHR-SIR-benchmark` | Comparison of SIR model and SIHR model for similar parameters | Models produce similar spread scenario but significantly differ in numbers of hospitalizations if fitted at the initial phase of disease spread
`NYC-data-preparation` | Model-independent analysis of NYC data | NYC data has seemingly no delay between identifying infections and hospitalizations; Looking at temporal variations, new admissions per day fluctuate around ~10-15%.
`NYC-social-distancing-fits-SIR` and `SIHR` | Fit analysis of SIR/SIHR model for NYC data | Data is best described if social distancing policies are fitted as well; to reliably extract social distancing fit parameters, a visible kink in new admissions per day should be visible; Fits after kink are consistent and allow meaningful extrapolations for 1-2 weeks.


## Install

Install dependencies using, for example, `pip` by running
```bash
pip install [--user] -r requirements.txt
```
To run the comparisons against Penn CHIME, you also have to install the `penn_chime` module from [github.com/CodeForPhilly/chime](https://github.com/CodeForPhilly/chime).

Scripts are supposed to be run from the repo root directory.


## Usage

The `NYC-predictions` notebook is a good start for learning how to use this module.
See the `doc/usage` notes for more details.


## Tests

Models implemented in this repo are compared against `penn_chime` in the `SIR-penn-chime-benchmark.ipynb ` notebook.
More tests for both models are implemented in the `tests` directory.
After installing `requirements-dev.txt`, you can run them with
```bash
pytest
```

## Contribute

Feel free to reach out and file issues for questions.
