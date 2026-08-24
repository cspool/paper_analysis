# **Ground Truth - Solution**

<Solution>

30 + 6 = 36

36 \* 6 = 216

216 / 8 = 27

</Solution>

```
Model Output - Solution
No feasible solution found.
Model Output - Solving Procedure
... ...
– |- Try 30 + 6 = 36. Add 36 to the number set. Current number set: [36, 8, 6], target: 27. Options for
choosing two numbers: [(36, 8), (36, 6), (8, 6)].
... ...
—- |- Pick two numbers (36, 6) (numbers left: [8]). Try possible operations.
—— |- Try 36 + 6 = 42. Add 42 to the number set. Current number set: [42, 8], target: 27, just two
numbers left.
——– |- Try 42 + 8 = 50. Evaluate 50 != 27, drop this branch.
——– |- Try 42 - 8 = 34. Evaluate 34 != 27, drop this branch.
——– |- Try 42 * 8 = 336. Evaluate 336 != 27, drop this branch.
——– |- Try 42 / 8 = 5.2. 5.2 is a decimal, drop this branch.
—— |- Try 36 - 6 = 30. Add 30 to the number set. Current number set: [30, 8], target: 27, just two
numbers left.
——– |- Try 30 + 8 = 38. Evaluate 38 != 27, drop this branch.
——– |- Try 30 - 8 = 22. Evaluate 22 != 27, drop this branch.
——– |- Try 30 * 8 = 240. Evaluate 240 != 27, drop this branch.
——– |- Try 30 / 8 = 3.7. 3.7 is a decimal, drop this branch.
—— |- Try 36 * 6 = 216. Evaluate 216 != 27, drop this branch.
... ...
```

### **G.4 Travel Planning**

Example [G.5](#page-40-0) shows an example of GPT-4o failing to update search states correctly of an 8K test set of the travel planning task. The model incorrectly drops the branch of [Mykonos, Zurich] after dropping one of the deeper child branches where the [Mykonos, Zurich] branch is part of the ground truth plan.

### <span id="page-40-0"></span>Example G.5: An example of search state update error for Travel Planning

