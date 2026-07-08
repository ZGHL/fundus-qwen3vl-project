# Stage-1.5 v3 Results (clean Adapter1-unseen, image-disjoint)

Test set: 1108 single-lesion samples.

## Baseline = Adapter1 vs Trained = Adapter1 + Stage1.5 v3 warm-start

| Lesion | metric | Adapter1 | Stage1.5 | Δ |
|---|---|---:|---:|---:|
| MA | parse_coverage | 0.968 | 0.949 | -0.019 |
| MA | f1 | 0.605 | 0.673 | 0.068 |
| MA | recall | 0.992 | 0.908 | -0.084 |
| MA | spec | 0.032 | 0.405 | 0.373 |
| MA | balanced_acc | 0.512 | 0.656 | 0.144 |
| MA | count_acc | None | None |  |
| MA | area_acc | None | None |  |
| HE | parse_coverage | 1.0 | 1.0 | 0.0 |
| HE | f1 | 0.703 | 0.706 | 0.003 |
| HE | recall | 0.881 | 0.706 | -0.175 |
| HE | spec | 0.595 | 0.81 | 0.215 |
| HE | balanced_acc | 0.738 | 0.758 | 0.02 |
| HE | count_acc | None | None |  |
| HE | area_acc | None | None |  |
| EX | parse_coverage | 1.0 | 0.993 | -0.007 |
| EX | f1 | 0.549 | 0.74 | 0.191 |
| EX | recall | 0.989 | 0.813 | -0.176 |
| EX | spec | 0.21 | 0.812 | 0.602 |
| EX | balanced_acc | 0.599 | 0.813 | 0.214 |
| EX | count_acc | None | None |  |
| EX | area_acc | None | None |  |
| SE | parse_coverage | 1.0 | 1.0 | 0.0 |
| SE | f1 | 0.256 | 0.344 | 0.088 |
| SE | recall | 1.0 | 0.95 | -0.05 |
| SE | spec | 0.017 | 0.397 | 0.38 |
| SE | balanced_acc | 0.508 | 0.673 | 0.165 |
| SE | count_acc | None | None |  |
| SE | area_acc | None | None |  |
| MACRO | parse_coverage | 0.992 | 0.986 | -0.006 |
| MACRO | f1 | 0.528 | 0.616 | 0.088 |
| MACRO | recall | 0.965 | 0.844 | -0.121 |
| MACRO | spec | 0.213 | 0.606 | 0.393 |
| MACRO | balanced_acc | 0.589 | 0.725 | 0.136 |
| MACRO | count_acc | None | None |  |
| MACRO | area_acc | None | None |  |

**Read:** parse failures count as classification errors. count_acc/area_acc are bucket accuracy on samples where both ground truth and prediction are present.
