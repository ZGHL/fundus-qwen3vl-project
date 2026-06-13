# Stage-1.5 v3 Results (clean Adapter1-unseen, image-disjoint)

Test set: 1108 single-lesion samples.

## Baseline = Adapter1 vs Trained = Adapter1 + Stage1.5 v3 warm-start

| Lesion | metric | Adapter1 | Stage1.5 | Δ |
|---|---|---:|---:|---:|
| MA | parse_coverage | 0.968 | 0.939 | -0.029 |
| MA | f1 | 0.605 | 0.574 | -0.031 |
| MA | recall | 0.992 | 0.924 | -0.068 |
| MA | spec | 0.032 | 0.025 | -0.007 |
| MA | balanced_acc | 0.512 | 0.475 | -0.037 |
| MA | count_acc | None | None |  |
| MA | area_acc | None | None |  |
| HE | parse_coverage | 1.0 | 0.986 | -0.014 |
| HE | f1 | 0.703 | 0.717 | 0.014 |
| HE | recall | 0.881 | 0.789 | -0.092 |
| HE | spec | 0.595 | 0.732 | 0.137 |
| HE | balanced_acc | 0.738 | 0.761 | 0.023 |
| HE | count_acc | None | None |  |
| HE | area_acc | None | None |  |
| EX | parse_coverage | 1.0 | 0.993 | -0.007 |
| EX | f1 | 0.549 | 0.642 | 0.093 |
| EX | recall | 0.989 | 0.945 | -0.044 |
| EX | spec | 0.21 | 0.511 | 0.301 |
| EX | balanced_acc | 0.599 | 0.728 | 0.129 |
| EX | count_acc | None | None |  |
| EX | area_acc | None | None |  |
| SE | parse_coverage | 1.0 | 1.0 | 0.0 |
| SE | f1 | 0.256 | 0.293 | 0.037 |
| SE | recall | 1.0 | 0.975 | -0.025 |
| SE | spec | 0.017 | 0.211 | 0.194 |
| SE | balanced_acc | 0.508 | 0.593 | 0.085 |
| SE | count_acc | None | None |  |
| SE | area_acc | None | None |  |
| MACRO | parse_coverage | 0.992 | 0.979 | -0.013 |
| MACRO | f1 | 0.528 | 0.557 | 0.029 |
| MACRO | recall | 0.965 | 0.908 | -0.057 |
| MACRO | spec | 0.213 | 0.37 | 0.157 |
| MACRO | balanced_acc | 0.589 | 0.639 | 0.05 |
| MACRO | count_acc | None | None |  |
| MACRO | area_acc | None | None |  |

**Read:** parse failures count as classification errors. count_acc/area_acc are bucket accuracy on samples where both ground truth and prediction are present.
