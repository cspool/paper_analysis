# F. Data Collection

We store RNG sequences either on-chip or off-chip depending on the available NVM capacity, as characterization requires up to 384 KB of storage in some cases ( $\S$ VI-B5). Characterizing a TRNG, as described in  $\S$ V-E1, requires  $2^n \cdot n$  bits of storage for n-bit outputs (over 16 GB for 32-bit generators) and takes roughly 136 years to collect considering 1 s per sample. <sup>5</sup> To make this feasible, we limit analysis to the 16 most significant bits, which are suitable for detecting weaknesses while requiring only 128 KB of storage, compatible with UISWaP NVM sizes. <sup>6</sup>

Devices with flash-based NVM often restrict writes to linesized operations. For these, we fill an array the size of the write line with the 16-bit random value, mask unused bits with ones, and write repeatedly to the same line until it is full before moving to the next available line. MSPM0 L-series Flash Controller also generates an Error Correction Code (ECC) during line writes, preventing rewrites to the same line before erasing it. In this case, we write to a shadow NVM address space that mirrors the actual NVM address space, but does not involve any ECC checks.

Devices with small NVMs, such as the ATSAML11E16A (64 KB), cannot store all samples, so we transmit data externally over UART. However, UART is slower than on-chip storage. Since we need to allow sufficient downtime between each power cycle (2 seconds on average), collecting  $2^{16}$  samples of 16-bit RNG sequences takes up to 55 hours in certain cases. Thus, we select the optimal strategy per device to minimize collection time and manage NVM constraints.

#### VI. EVALUATION

All intermittent computing devices provide the abstraction of random number generation, even when they lack the capability to produce true randomness. In this section, we assess the support for cryptographic-grade random number generation available to intermittent systems using our representative device suite. Our evaluation addresses the following questions:

- What type of RNGs exist in intermittent platforms?
- How do these RNGs perform under intermittent power?
- How do these RNGs perform in an attacker-controlled environment?
- How do devices with PRNGs resist software-level attack?
- Which devices are viable foundations for future secure intermittent computing systems?

