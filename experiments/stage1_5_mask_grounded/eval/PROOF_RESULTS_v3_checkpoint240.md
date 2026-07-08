# Stage-1.5 v3 Results (clean Adapter1-unseen, image-disjoint)

Test set: 1108 single-lesion samples.

## Baseline = Adapter1 vs Trained = Adapter1 + Stage1.5 v3 warm-start

| Lesion | metric | Adapter1 | Stage1.5 | Δ |
|---|---|---:|---:|---:|
| MA | parse_coverage | 0.968 | 0.946 | -0.022 |
| MA | f1 | 0.605 | 0.598 | -0.007 |
| MA | recall | 0.992 | 0.908 | -0.084 |
| MA | spec | 0.032 | 0.152 | 0.12 |
| MA | balanced_acc | 0.512 | 0.53 | 0.018 |
| MA | count_acc | None | None |  |
| MA | area_acc | None | None |  |
| HE | parse_coverage | 1.0 | 0.993 | -0.007 |
| HE | f1 | 0.703 | 0.731 | 0.028 |
| HE | recall | 0.881 | 0.798 | -0.083 |
| HE | spec | 0.595 | 0.75 | 0.155 |
| HE | balanced_acc | 0.738 | 0.774 | 0.036 |
| HE | count_acc | None | None |  |
| HE | area_acc | None | None |  |
| EX | parse_coverage | 1.0 | 1.0 | 0.0 |
| EX | f1 | 0.549 | 0.714 | 0.165 |
| EX | recall | 0.989 | 0.879 | -0.11 |
| EX | spec | 0.21 | 0.715 | 0.505 |
| EX | balanced_acc | 0.599 | 0.797 | 0.198 |
| EX | count_acc | None | None |  |
| EX | area_acc | None | None |  |
| SE | parse_coverage | 1.0 | 1.0 | 0.0 |
| SE | f1 | 0.256 | 0.333 | 0.077 |
| SE | recall | 1.0 | 0.925 | -0.075 |
| SE | spec | 0.017 | 0.388 | 0.371 |
| SE | balanced_acc | 0.508 | 0.657 | 0.149 |
| SE | count_acc | None | None |  |
| SE | area_acc | None | None |  |
| MACRO | parse_coverage | 0.992 | 0.985 | -0.007 |
| MACRO | f1 | 0.528 | 0.594 | 0.066 |
| MACRO | recall | 0.965 | 0.877 | -0.088 |
| MACRO | spec | 0.213 | 0.501 | 0.288 |
| MACRO | balanced_acc | 0.589 | 0.689 | 0.1 |
| MACRO | count_acc | None | None |  |
| MACRO | area_acc | None | None |  |

**Read:** parse failures count as classification errors. count_acc/area_acc are bucket accuracy on samples where both ground truth and prediction are present.
