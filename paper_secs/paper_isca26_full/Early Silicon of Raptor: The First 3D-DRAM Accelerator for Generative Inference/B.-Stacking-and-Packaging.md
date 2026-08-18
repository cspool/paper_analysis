# *B. Stacking and Packaging*

Each accelerator card integrates up to four identical MCMs. Each MCM is built on a 9-4-9 organic substrate providing sufficient routing density for high-speed I/O at a ∼422 W power target and junction temperatures up to 105◦C. Within each MCM, every chiplet employs face-to-face 3D stacking: the logic die (TSMC N4P) bonds directly to a DRAM die through a 36 µm-pitch µbump array, exposing wide, lowlatency channels into PHY blocks on the logic die.

The 3D-DRAM die organizes its 840 banks into chains that support redundancy and per-bank configurability (§IV-C and §IV-E); these banks are mapped onto the gang/slice hierarchy to form balanced channels. The logic-DRAM stack connects via mixed-pitch C4 bumps (minimum 110 µm) to a 3D CoWoS interposer, which fans out signals and power to the substrate and to eight on-package LPDDR5X-9600 devices (561-ball, qualified to 95◦C, ∼3.3 W each). These devices provide 128 GB per MCM as a secondary memory tier for model parameters and KV cache that do not fit in 3D-DRAM.

