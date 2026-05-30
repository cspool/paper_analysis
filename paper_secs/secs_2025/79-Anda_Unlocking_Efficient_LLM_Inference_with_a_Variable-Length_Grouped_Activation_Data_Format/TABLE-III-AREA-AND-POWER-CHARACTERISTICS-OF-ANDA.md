# TABLE III AREA AND POWER CHARACTERISTICS OF ANDA

| Component         | Setup                        | Area [mm <sup>2</sup> ] | Power [mW]      |
|-------------------|------------------------------|-------------------------|-----------------|
| MXU               | 16×16 APUs                   | 0.41 (18.89%)           | 54.34 (66.94%)  |
| BPC               | 16 Lanes                     | 0.07 (3.23%)            | 1.06 (1.31%)    |
| Vector Unit       | 64 FPUs                      | 0.05 (2.30%)            | 0.87 (1.07%)    |
| Activation Buffer | 1MB (Mant.) + 0.125MB (Exp.) | 0.87 (40.09%)           | 16.94 (20.87%)  |
| Weight Buffer     | 1MB                          | 0.80 (36.87%)           | 7.96 (9.81%)    |
| Others            | Top controller               | 0.01 (0.46%)            | 0.01 (0.00%)    |
| Total             |                              | 2.17 (100.00%)          | 81.18 (100.00%) |

Table III presents the area breakdown and power distribution. Operating at 285 MHz and 0.8 V, Anda occupies 2.17 mm² with 81.18 mW power consumption. The MXU, serving as the core computing component of the Anda architecture, consumes 66.94% of the total power despite occupying only 18.89% of the area. The BPC unit, which enables efficient online compression from the full-precision FP outputs to the Anda format, costs a small portion of the total area (3.23%) and power consumption (1.31%). On-chip SRAM is the primary area contributor, with the activation buffer and weight buffer accounting for 40.09% and 36.87% of the total area, respectively. Their power consumption ratios are relatively low at 20.87% and 9.81% because of efficient data reuse within the Anda system.

