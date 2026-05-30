# V. METHODOLOGY

We evaluate ACS-SW on a real hardware setup with an Intel Core i7 11700K CPU (Table III) and an NVIDIA RTX3060

| Window<br>size | Number of RW-segments | Dependency check time |
|----------------|-----------------------|-----------------------|
| 16             | 6<br>10               | 410ns<br>700ns        |
| 32.            | 6                     | 510ns                 |
| 32             | 10                    | 1640ns                |

TABLE II: Dependency checking overhead analysis GPU (Table IV). We model ACS-HW on GPUs using the Accel-Sim simulator [64], configured with parameters of RTX3070 (Table V). We use AccelWattch [64] to model GPU power. We choose a scheduling window size of 32.

| CPU 3.6GHz, OOO 4-wide dispatch window, 32 entry LSQ    |
|---------------------------------------------------------|
| L1D + L1I Cache 32KB, 4 way LRU, 1 cycle; 64 Byte line; |
| L2 Cache 256KB, 8 way LRU, 4 cycle; 64 Byte line;       |
| L3 Cache 1MB, 16 way LRU, 20 cycle; 64 Byte line;       |
| DRAM 2-channel; 16-bank; open-row policy, 4GB DDR4      |

TABLE III: CPU system configuration

| Shader core 28 SMs, 1.3GHz; 2 schedulers per SM             |
|-------------------------------------------------------------|
| SM Resources 32768 Registers, 32KB Shared memory, 128KB L1D |
| <b>DRAM</b> 2-channel; 16-bank; open-row policy, 12GB DDR4  |

TABLE IV: GPU system configuration

| Shader core 46 SMs, 1.4GHz; 4 schedulers per SM             |
|-------------------------------------------------------------|
| SM Resources 32768 Registers, 32KB Shared memory, 128KB L1D |
| <b>DRAM</b> 2-channel; 16-bank; open-row policy, 16GB DDR4  |

TABLE V: Simulated GPU configuration

