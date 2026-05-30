# *D. Area and Energy Efficiency Evaluation*

We complete the RTL synthesis, placement, and routing for MECLA processor with 28nm 1P8M CMOS for the chip area. Then we use Synopsys PrimeTime and VCS to perform post simulation and power analysis with actual running data for precise energy evaluation. Figure 13 shows its post layout, and Figure 14 shows its power and area breakdown. MECLA has a die area of 22.02 mm<sup>2</sup>. The PE array, on-chip buffer, and scaling accumulator take up 26%, 31%, and 20% of the area, respectively. During operation at 1GHz, the processor's average power is 2.87W, with the PE array and memory consuming 52.3% and 34.4% of the energy. In comparison, the latest A17 Pro mobile SoC with 3nm technology consumes 11-14W of power when running heavy tasks. Only 2.56% of energy overhead (normalized to 3nm technology) would be required if MECLA processor is embedded in the mobile SoC to enable efficient LLM inference.

TABLE III COMPARISON OF ACCELERATION EFFECT AND SPECIFICATIONS WITH STATE-OF-THE-ART ACCELERATORS

|                               | SpAtten1/8 [77]             | Sanger [46]                 | FACT [55]                | MECLA                  |
|-------------------------------|-----------------------------|-----------------------------|--------------------------|------------------------|
| Accelerate                    | Decoder<br>(attention only) | Encoder<br>(Attention only) | Encoder<br>(whole model) | LLM1)<br>(whole model) |
| Optimization2)                | C Only                      | C Only                      | C Only                   | C&M                    |
| Technology (nm)               | 40                          | 55                          | 28                       | 28                     |
| Area (mm2)                    | 1.55                        | 16.9                        | 6.03                     | 22.02                  |
| Throughput (GOPS)             | 360                         | 529                         | 928                      | 14008                  |
| Energy Efficiency<br>(GOPS/W) | 382                         | 192                         | 4388                     | 7088                   |

- 1) MECLA's acceleration works for both encoder and decoder, and performs better on the latter.
- 2) C: computation. M: memory access.

Figure 12 shows the energy efficiency improvement of MECLA processor and V100 GPU compared to naive implementation baseline. On geometric average, using standard and aggressive SSMP optimization for inference with GPU improves the energy consumption by an average of 1.83× / 2.01×. Combined with the specially designed circuits in MECLA, the efficiency improvement is improved by 34.3×/ 48.6×, achieving 5.01 / 7.09 TOPS/W on average. The peak energy efficiency reaches 7.43 and 9.71TOPS/W at 1.0V, 1GHz, using MECLA standard and aggressive. Such high improvements come from two main factors. For memory concerns, MECLA saves 80.4%-89.5% weight parameter access, which greatly reduces the memory footprint and alleviates the bandwidth bounding. For computation concerns, the PSum reuse reduces 76.6%-78.7% computation with the special PE array design.

