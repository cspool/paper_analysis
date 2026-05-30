# A.7 Experiment customization

Key parameters can be modified in source files:

- QoS tiers: sequence.py (line 11)
- Hybrid prioritization : sequence.py (line 79)
- Scheduling logic: deadline\_scheduler.py

Refer to README for implementation details and customization guide.

## A.8 Notes

GPU clock locking (described in README) is essential for reproducible measurements. The current artifact uses a smaller number of requests for manageable runtime; the original paper evaluated on significantly larger request volumes for stronger statistical significance. This may lead to variations in reproducing exact numerical results, though relative performance trends and conclusions remain consistent.