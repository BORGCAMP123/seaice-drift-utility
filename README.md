# Sea-ice drift detail and optical-state reconstruction

This repository contains the event-level data tables and deterministic scripts used to reproduce the statistics, main figures, and Supplementary Information of the accompanying manuscript. The inferential units are 29 informative independent event/resource clusters. Raw VIIRS and SID-CV products are not redistributed.

## Directory guide

- `metadata/event_inventory.csv`: eligible cases, event membership, timing, sample role, and source/target/drift provenance.
- `tables/transport_evaluation/`: explicit transport relative to persistence.
- `tables/drift_scale_analysis/`: controlled hierarchy, adjacent-band utility, response shape, and diagnostic variables.
- `tables/external_validation/`: discovery-selected rules and independent validation results.
- `tables/prediction_error_decomposition/`: perturbation, alignment, penalty, and identity checks.
- `tables/robustness_analysis/`: sample-group summaries.
- `scripts/`: validation and deterministic figure/Supplement generators.
- `figures/`: final main figures (PDF and SVG).
- `supplementary/`: Supplementary Information, tables, and figures.
- `environment/`: Python dependency specifications.

## Reproduce the paper outputs

Create the environment from `environment/conda.yml` or install `environment/requirements.txt`, then run from the repository root:

```powershell
python scripts/validate_results.py
python scripts/plot_study_design.py
python scripts/plot_main_figures.py
python scripts/build_supplementary_information.py
```

The scripts read the included event-level tables and write PDF/SVG/PNG figures, Supplement tables, and source records. See `reproducibility.md` for expected checks and `data_acquisition.md` for product provenance. These scripts reproduce the published event-level analysis; they do not implement raw-product preprocessing.

## Scientific scope and data rights

SID-CV is retrieved displacement used for observed-interval reconstruction, not operational forecast drift. Gaussian FWHM is a filter parameter, not effective spatial resolution. Retrospective best levels and residual alignment are diagnostic, not deployment rules. The tested simple rules had limited transfer across independent events.

Raw products remain with their providers. A source-code license and public repository address require author approval before publication; no license is implied by this directory.
