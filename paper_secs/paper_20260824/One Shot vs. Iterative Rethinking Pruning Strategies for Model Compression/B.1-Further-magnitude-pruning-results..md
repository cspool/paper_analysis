# *B.1 Further magnitude-pruning results.*

Figure 2 presents further comparison of one-shot and iterative pruning across various network architectures and vision datasets. The iterative pruning comes in two types: constant and geometric.

## C Early stopping.

In this section, we provide the detailed algorithm for early stopping performed in this paper. We use this algorithm for both one-shot and iterative geometric and iterative constant pruning.

### Algorithm 1 Early Stopping Check

Note: This code assumes that a lower metric value indicates better performance (e.g., loss). Otherwise, if a higher metric value is better (e.g., accuracy) the code is run with a reversed comparison.

```
1: procedure EARLYSTOP(metric_value : f loat)
2: if metric_value < self.best_metric_value then
3: self.best_metric_value ← metric_value
4: self.counter ← 0
5: else if metric_value > (self.best_metric_value +
   self.min_delta) then
6: self.counter ← self.counter + 1
7: end if
8: if self.counter ≥ self.patience then
9: return True
10: end if
11: return False
12: end procedure
```

