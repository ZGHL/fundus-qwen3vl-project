# Stage-1.5 v3 Results (clean Adapter1-unseen, image-disjoint)

Test set: 1108 single-lesion samples.

## Baseline = Adapter1 vs Trained = Adapter1 + Stage1.5 v3 warm-start

| Lesion | metric | Adapter1 | Stage1.5 | Δ |
|---|---|---:|---:|---:|
| MA | parse_coverage | 0.968 | 0.935 | -0.033 |
| MA | f1 | 0.605 | 0.6 | -0.005 |
| MA | recall | 0.992 | 0.933 | -0.059 |
| MA | spec | 0.032 | 0.114 | 0.082 |
| MA | balanced_acc | 0.512 | 0.523 | 0.011 |
| MA | count_acc | None | None |  |
| MA | area_acc | None | None |  |
| HE | parse_coverage | 1.0 | 0.986 | -0.014 |
| HE | f1 | 0.703 | 0.702 | -0.001 |
| HE | recall | 0.881 | 0.725 | -0.156 |
| HE | spec | 0.595 | 0.78 | 0.185 |
| HE | balanced_acc | 0.738 | 0.752 | 0.014 |
| HE | count_acc | None | None |  |
| HE | area_acc | None | None |  |
| EX | parse_coverage | 1.0 | 0.996 | -0.004 |
| EX | f1 | 0.549 | 0.692 | 0.143 |
| EX | recall | 0.989 | 0.912 | -0.077 |
| EX | spec | 0.21 | 0.645 | 0.435 |
| EX | balanced_acc | 0.599 | 0.779 | 0.18 |
| EX | count_acc | None | None |  |
| EX | area_acc | None | None |  |
| SE | parse_coverage | 1.0 | 1.0 | 0.0 |
| SE | f1 | 0.256 | 0.304 | 0.048 |
| SE | recall | 1.0 | 1.0 | 0.0 |
| SE | spec | 0.017 | 0.228 | 0.211 |
| SE | balanced_acc | 0.508 | 0.614 | 0.106 |
| SE | count_acc | None | None |  |
| SE | area_acc | None | None |  |
| MACRO | parse_coverage | 0.992 | 0.979 | -0.013 |
| MACRO | f1 | 0.528 | 0.575 | 0.047 |
| MACRO | recall | 0.965 | 0.892 | -0.073 |
| MACRO | spec | 0.213 | 0.442 | 0.229 |
| MACRO | balanced_acc | 0.589 | 0.667 | 0.078 |
| MACRO | count_acc | None | None |  |
| MACRO | area_acc | None | None |  |

**Read:** parse failures count as classification errors. count_acc/area_acc are bucket accuracy on samples where both ground truth and prediction are present.
