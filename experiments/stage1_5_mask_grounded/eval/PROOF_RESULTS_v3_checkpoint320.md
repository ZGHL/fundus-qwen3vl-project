# Stage-1.5 v3 Results (clean Adapter1-unseen, image-disjoint)

Test set: 1108 single-lesion samples.

## Baseline = Adapter1 vs Trained = Adapter1 + Stage1.5 v3 warm-start

| Lesion | metric | Adapter1 | Stage1.5 | Δ |
|---|---|---:|---:|---:|
| MA | parse_coverage | 0.968 | 0.892 | -0.076 |
| MA | f1 | 0.605 | 0.585 | -0.02 |
| MA | recall | 0.992 | 0.882 | -0.11 |
| MA | spec | 0.032 | 0.146 | 0.114 |
| MA | balanced_acc | 0.512 | 0.514 | 0.002 |
| MA | count_acc | None | None |  |
| MA | area_acc | None | None |  |
| HE | parse_coverage | 1.0 | 0.993 | -0.007 |
| HE | f1 | 0.703 | 0.73 | 0.027 |
| HE | recall | 0.881 | 0.78 | -0.101 |
| HE | spec | 0.595 | 0.768 | 0.173 |
| HE | balanced_acc | 0.738 | 0.774 | 0.036 |
| HE | count_acc | None | None |  |
| HE | area_acc | None | None |  |
| EX | parse_coverage | 1.0 | 0.993 | -0.007 |
| EX | f1 | 0.549 | 0.716 | 0.167 |
| EX | recall | 0.989 | 0.846 | -0.143 |
| EX | spec | 0.21 | 0.747 | 0.537 |
| EX | balanced_acc | 0.599 | 0.797 | 0.198 |
| EX | count_acc | None | None |  |
| EX | area_acc | None | None |  |
| SE | parse_coverage | 1.0 | 1.0 | 0.0 |
| SE | f1 | 0.256 | 0.338 | 0.082 |
| SE | recall | 1.0 | 0.975 | -0.025 |
| SE | spec | 0.017 | 0.359 | 0.342 |
| SE | balanced_acc | 0.508 | 0.667 | 0.159 |
| SE | count_acc | None | None |  |
| SE | area_acc | None | None |  |
| MACRO | parse_coverage | 0.992 | 0.969 | -0.023 |
| MACRO | f1 | 0.528 | 0.592 | 0.064 |
| MACRO | recall | 0.965 | 0.871 | -0.094 |
| MACRO | spec | 0.213 | 0.505 | 0.292 |
| MACRO | balanced_acc | 0.589 | 0.688 | 0.099 |
| MACRO | count_acc | None | None |  |
| MACRO | area_acc | None | None |  |

**Read:** parse failures count as classification errors. count_acc/area_acc are bucket accuracy on samples where both ground truth and prediction are present.
