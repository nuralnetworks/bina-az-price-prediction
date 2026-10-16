# Contribution Report

## Team
| Member (Surname, Name) | GitHub handle | Main responsibilities |
|------------------------|---------------|-----------------------|
| Shukurlu, Nural | nuralnetworks | Everything: data pipeline, both from-scratch models, benchmarking, report, slides |

## Per-member summary
**Shukurlu, Nural —** I did this project solo, end to end. I designed the cleaning pipeline in `src/data_prep.py` (deduplication, Azerbaijani field parsing, leakage removal, tier label, splits), implemented the decision tree in `src/decision_tree.py` and the Pegasos SVM with random Fourier features in `src/svm.py`, wrote the metrics in `src/evaluate.py` and the full benchmark in `src/run_all.py`, then ran the sweeps, analysed the errors, and wrote `report/report.tex` and `presentation/slides.tex` from the resulting numbers and figures.

## Rough effort split

| Member | Approx. % |
|--------|-----------|
| Shukurlu, Nural | 100% |
