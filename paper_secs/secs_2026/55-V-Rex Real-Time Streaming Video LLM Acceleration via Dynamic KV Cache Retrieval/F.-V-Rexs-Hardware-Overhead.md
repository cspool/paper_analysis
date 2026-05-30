# F. V-Rex's Hardware Overhead

**Power and Area.** Table III summarizes the power consumption and area breakdown for V-Rex equipped with a single core. A single V-Rex core occupies 1.89 mm<sup>2</sup> and consumes 2.61 W, equipped with on-chip memory of 384 KB for LXE and 20.125 KB for DRE. When scaled to larger configurations, the area of V-Rex<sup>8</sup> is 15.12 mm<sup>2</sup>, which is substantially smaller than the AGX Orin GPU (200 mm<sup>2</sup>). Notably, V-Rex<sup>48</sup>

TABLE III BREAKDOWN OF AREA AND POWER

|     | Hardware Components |      | Area[mm2] |         | Power [mW] |
|-----|---------------------|------|-----------|---------|------------|
|     | DPE                 | 1.37 | 72.79%    | 2311.39 | 88.58%     |
| LXE | VPE                 | 0.14 | 7.18%     | 122.06  | 4.68%      |
|     | On-chip Memory      | 0.34 | 17.98%    | 118.94  | 4.56%      |
| DRE | KVPU -<br>HCU       | 0.01 | 0.28%     | 2.99    | 0.11%      |
|     | KVPU -<br>WTU       | 0.02 | 1.23%     | 39.04   | 1.49%      |
|     | KVMU                | 0.01 | 0.53%     | 15.01   | 0.58%      |
|     | Total               | 1.89 | 100%      | 2609.43 | 100%       |

occupies 90.57 mm<sup>2</sup> , considerably less than the A100 GPU (826 mm<sup>2</sup> ). Including overall system power, V-Rex<sup>8</sup> consumes 35 W, achieving 11.4% lower power consumption than the AGX Orin GPU (40 W), while V-Rex<sup>48</sup> consumes 203.68 W, demonstrating 32.1% lower power consumption than the A100 GPU (300 W), as detailed in Table I. The additional hardware overhead of DRE is minimal, accounting for only 2.4% of the chip's total power and 2.0% of the total area, which can be attributed to the effective KV cache retrieval algorithm. Its compact design enables efficient integration with any existing GPUs, NPUs, and LLM accelerators.

