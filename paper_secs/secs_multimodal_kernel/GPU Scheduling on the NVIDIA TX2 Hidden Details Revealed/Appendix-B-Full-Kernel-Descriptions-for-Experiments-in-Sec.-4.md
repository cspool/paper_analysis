# Appendix B: Full Kernel Descriptions for Experiments in Sec. 4

In this section, we present the configuration data for the experiments in Sec. 4. None of these experiments use shared

|        | Launch      | Start    |              |          | # Thread  |
|--------|-------------|----------|--------------|----------|-----------|
| Kernel | Info        | Time (s) | Duration (s) | # Blocks | per Block |
| K1     | Stream S1   | 0.0      | 1.0          | 6        | 768       |
| K2     | NULL Stream | 0.2      | 1.0          | 1        | 1,024     |
| K3     | Stream S2   | 0.2      | 1.0          | 4        | 256       |
| K4     | Stream S2   | 0.4      | 1.0          | 4        | 256       |
| K5     | NULL Stream | 0.6      | 1.0          | 1        | 1,024     |
| K6     | Stream S3   | 0.8      | 1.0          | 2        | 256       |

Table 2: Details of kernels used in the NULL-stream scheduling experiment in Fig. 5.

|        |                           | Start    |              |          | # Thread  |
|--------|---------------------------|----------|--------------|----------|-----------|
| Kernel | Launch Info               | Time (s) | Duration (s) | # Blocks | per Block |
| K1     | Stream S1 (low priority)  | 0.0      | 0.5          | 8        | 1,024     |
| K2     | Stream S2 (high priority) | 0.2      | 0.5          | 16       | 1,024     |
| K3     | Stream S3 (high priority) | 0.5      | 0.5          | 16       | 1,024     |

Table 3: Details of kernels used in the priority-stream scheduling experiment in Fig. 6.

|        |                                  | Start    |              |          | # Thread  |
|--------|----------------------------------|----------|--------------|----------|-----------|
| Kernel | Launch Info                      | Time (s) | Duration (s) | # Blocks | per Block |
| K1     | Stream S1 (low priority)         | 0.0      | 0.5          | 8        | 1,024     |
| K2     | Stream S2 (unspecified priority) | 0.2      | 0.5          | 8        | 1,024     |
| K3     | Stream S3 (high priority)        | 0.3      | 0.5          | 8        | 1,024     |
| K4     | Stream S4 (low priority)         | 1.2      | 0.5          | 8        | 1,024     |

Table 4: Details of kernels used in the priority-stream scheduling experiment in Fig. 7.

|        |                           | Start    |              |          | # Thread  |
|--------|---------------------------|----------|--------------|----------|-----------|
| Kernel | Launch Info               | Time (s) | Duration (s) | # Blocks | per Block |
| K1     | Stream S1 (low priority)  | 0.0      | 1.0          | 1        | 512       |
| K2     | Stream S2 (low priority)  | 0.1      | 1.0          | 1        | 512       |
| K3     | Stream S3 (low priority)  | 0.2      | 1.0          | 1        | 512       |
| K4     | Stream S4 (low priority)  | 0.3      | 1.0          | 1        | 512       |
| K5     | Stream S5 (low priority)  | 0.4      | 1.0          | 1        | 512       |
| K6     | Stream S6 (low priority)  | 0.5      | 1.0          | 1        | 512       |
| K7     | Stream S7 (low priority)  | 0.6      | 1.0          | 1        | 512       |
| K8     | Stream S8 (high priority) | 0.65     | 0.5          | 1        | 1,024     |
| K9     | Stream S9 (low priority)  | 0.7      | 1.0          | 1        | 512       |

Table 5: Details of kernels used in the priority-stream scheduling experiment in Fig. 8.