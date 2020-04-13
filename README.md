# COVID-19 Analysis of NYC Data

Notebooks in this repository provide Python tools for fitting variations of the [SIR model](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIR_model) to COVID-19 data in a statistical context.
This allows us to extract parameters like social distancing policies and initial infections with an estimated uncertainty.
Models are applied to data provided by the [NYC Department of Health and Mental Hygiene (DOHMH)](https://github.com/nychealth/coronavirus-data).

## Details



## Disclaimer

I do not have a background in epidemiology or related fields.
For now, the analysis was **not cross-checked and reviewed by an expert in the field**.
While I am confident that predictions are stable in a time-frame for 1-2 weeks from a data-driven point of view, I might have missed important details specific to this problem.
If you are familiar with this field, please reach out and provide feedback.

## Conclusion


## Room for improvements


## Data

Notebooks in this repo make use of the  [NYC Department of Health and Mental Hygiene (DOHMH) repo data](https://github.com/nychealth/coronavirus-data).
In particular the `case-hosp-death.csv` is used.

## Model

This repository implements the regular SIR model and 3 variations.
* Standard SIR
* SIR model with time-dependent `beta` parameter (`beta * (1 - logistic)`)
* "SIHR" model which is explained below
* "SIHR" model with time-dependent `beta` parameter (`beta * (1 - logistic)`)

### The SIHR model

The SIHR model is a variation of the SIR model which explicitly encodes the number of hospitalizations as a new compartment.

## Content

### Computation

Folder | Description
---|---
`models` | Implementation of the SIR and SIHR model including wrappers which simplify fitting
`utils` | Utility functions for loading data and plotting fits

### Analysis

File | Description | Conclusion
---|---|---
`SIR-penn-chime-benchmark.ipynb` | Comparison of SIR model in this repository against `penn_chime` | Both modules agree with and without social distancing measures.
`SIHR-SIR-benchmark.ipynb` | Comparison of SIR model and SIHR model for similar parameters | Models produce similar spread scenario but significantly differ in numbers of hospitalizations if fitted at the initial phase of disease spread
`NYC-data-preparation.ipynb` | Model-independent analysis of NYC data | NYC data has seemingly no delay between identifying infections and hospitalizations; Looking at temporal variations, new admissions per day fluctuate around ~10-15%.
`NYC-social-distancing-fits-SIR.ipynb` and `NYC-social-distancing-fits-SIHR.ipynb` | Fit analysis of SIHR model for NYC data | Fit analysis of SIR/SIHR model for NYC data | Data is best described if social distancing policies are fitted as well; to reliably extract social distancing fit parameters, a visible kink in new admissions per day should be visible; Fits after kink are consistent and allow meaningful extrapolations for 1-2 weeks.



## Install

Install dependencies using, for example, `pip` by running
```bash
pip install [--user] -r requirements.txt
```
To run the comparisons against Penn CHIME, you also have to install the `penn_chime` module from [github.com/CodeForPhilly/chime](https://github.com/CodeForPhilly/chime).

Scripts are supposed to be run from the repo root directory.


## Usage



## Tests

Models implemented in this repo are compared against `penn_chime` in the `SIR-penn-chime-benchmark.ipynb ` notebook.

## Contribute

Feel free to reach out and file issues for questions.
