# **B** Test-Time-Scaling Strategies

We ran the A\*-Thought evaluation on MATH500 and LiveCodeBench, this time using a best-of-N strategy (with N=8) to calculate pass@k. The findings indicate that A\*-Thought and test-time-scaling strategies can be used in combination, the experiment results are shown in Table 5.

<span id="page-14-0"></span>Table 5: Experimental results on A\*-Thought with test-time-scaling strategies.

| Methods      |                     | MAT    | H500             |           |        | LiveCoo | deBench |        |
|--------------|---------------------|--------|------------------|-----------|--------|---------|---------|--------|
| 1,10110415   | pass@1              | pass@2 | pass@4           | pass@8    | pass@1 | pass@2  | pass@4  | pass@8 |
|              |                     |        | Budget:          | 512 Token | s      |         |         |        |
| QwQ-32B      | 10.8                | 17.2   | 25.9             | 34.4      | 0.0    | 0.0     | 0.0     | 0.0    |
| + A*-Thought | 33.2                | 47.4   | 60.7             | 70.2      | 4.5    | 7.0     | 12.5    | 20.8   |
|              | Budget: 1024 Tokens |        |                  |           |        |         |         |        |
| QwQ-32B      | 16.6                | 29.6   | 42.1             | 53.2      | 0.0    | 0.1     | 0.1     | 0.3    |
| + A*-Thought | 50.8                | 65.4   | 75.8             | 81.8      | 11.8   | 20.4    | 31.9    | 44.8   |
|              |                     |        | <b>Budget: 2</b> | 048 Token | ıs     |         |         |        |
| QwQ-32B      | 51.2                | 59.8   | 67.8             | 74.8      | 2.1    | 3.4     | 5.1     | 7.0    |
| + A*-Thought | 69.2                | 79.2   | 85.4             | 89.0      | 24.5   | 34.0    | 45.2    | 56.5   |
|              | Budget: 4096 Tokens |        |                  |           |        |         |         |        |
| QwQ-32B      | 75.4                | 80.2   | 84.0             | 86.8      | 12.4   | 17.7    | 23.0    | 28.0   |
| + A*-Thought | 78.8                | 86.8   | 91.1             | 93.2      | 39.0   | 50.9    | 60.9    | 68.8   |

