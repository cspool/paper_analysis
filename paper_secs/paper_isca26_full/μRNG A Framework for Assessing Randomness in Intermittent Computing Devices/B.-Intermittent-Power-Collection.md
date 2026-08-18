# *B. Intermittent Power Collection*

To assess the impact of intermittent operation on random number generation, we power cycle the target, collecting the most significant 16 bits of the first RNG output each cycle. We perform this collection across 65,536 power cycles,

![](_page_4_Picture_8.jpeg)

![](_page_4_Picture_9.jpeg)

Fig. 4: Left: TE-123H temperature/humidity chamber. Right: MSPM0L2228 evaluation board in an airtight bag inside the thermal chamber with soldered connections going out.

concatenating the results to form a single 128 KB bit stream that we pass on to statistical testing. Each power cycle includes sufficient downtime (2 seconds) to discharge residual conductance and prevent carry-over from prior power-on periods.

To systematically power cycle the devices across a large number of trials and vary their supply voltage, we create a power controller as depicted in Figure 3. The power controller consists of a single pole single throw TLP222AF solid state relay [19] along with a MCP4725 Digital-to-Analog Converter [34] (DAC) on its power plane, which transforms the input voltage to the desired output voltage. We use a Raspberry Pi 3B to send control signals to the relay and supply its power plane with a steady 5V. The Raspberry Pi communicates with the DAC via I2C protocol to control the output voltage, which ranges from 0V to 5V. The Raspberry Pi also serves as the external storage device when the MCU on-board storage is insufficient. Finally, we execute a script on the Raspberry Pi to automate the scheduled power on and off cycles of the Device Under Test (DUT).

## *C. Thermal Control Configuration*

To stress test RNGs under extreme environmental conditions, we follow the device-specific operating ranges, listed in Table I. For extreme thermal conditions, we consider the industrial operation range of -40 C to +85 C. Although it is tempting to go beyond this range, most devices fail to operate outside of it as solder weakens if subjected to temperatures above +85 C for extended periods of time. However, during our experiments, we observe that some devices withstand temperatures as low as -68 C. For precise control of the testbed environment for extended periods, we use the TE-123H temperature/humidity chamber [83] shown in Figure 4. This chamber features a thermal range of -68 C to +175 C and has the ability to reliably maintain <25% humidity at room temperature and above. We place our DUT inside the chamber, connected to the power control setup fixed on the outer wall of the chamber.<sup>4</sup> During our experimental trials, we observe that cooling the DUT to sub-freezing temperatures

<sup>4</sup>The operating ranges of individual components in our power control setup interfere with the DUT's operating range, so we opt to place it outside the chamber.

| Device<br>Family            | Processor<br>Core | Max Clock<br>(MHz) | NVM<br>Type | NVM<br>Size (KB) | Min NVM<br>Write Size (B) | SRAM<br>Size (KB) | Operating Volt. (V) | Operating Temp. (C) |
|-----------------------------|-------------------|--------------------|-------------|------------------|---------------------------|-------------------|---------------------|---------------------|
| (1) FE310-G002 [75]         | RISC-V            | 320                | Flash       | 4096*            | 1                         | 16                | 1.8 - 3.3           | -40 / +85           |
| (44) MSP430FR59x/69x [86]   | MSP430X           | 16                 | FRAM        | 32 - 256         | 1                         | 1 - 8             | 1.8 - 3.6           | -68 / +85           |
| (23) MSPM0 L-Series [88]    | Cortex-M0+        | 32                 | Flash       | 8 - 256          | 8                         | 2 - 32            | 1.62 - 3.6          | -68 / +85           |
| (15) SAM D21 [56]           | Cortex-M0+        | 48                 | Flash       | 32 - 256         | 64                        | 4 - 32            | 1.62 - 3.63         | -40 / +125          |
| (27) SAM L10/L11 [57]       | Cortex-M23        | 32 - 48            | Flash       | 16 - 512         | 64                        | 4 - 64            | 1.62 - 3.63         | -68 / +215          |
| ( <b>51</b> ) TM4C123x [87] | Cortex-M4         | 80                 | Flash       | 32 - 256         | 4                         | 12 - 32           | 1.7 - 3.3           | -40 / +105          |
| (14) MSP432Px [84]          | Cortex-M4         | 24 - 48            | Flash       | 128 - 2048       | 1                         | 32 - 256          | 1.62 - 3.7          | -68 / +85           |
| (7) Apollo3 [2]             | Cortex-M4         | 96                 | Flash       | 1024 - 2048      | 16                        | 384 - 768         | 1.75 - 3.63         | -40 / +85           |
| (7) Apollo4 [3]             | Cortex-M4         | 192                | MRAM        | 2048             | 16                        | 1024 - 2816       | 1.71 - 2.2          | -40 / +85           |

TABLE I: Device family specifications for MCUs we evaluate. Operating ranges reflect observed tolerances during experiments, not necessarily documented limits. \*Size determined by external flash chip. The number before each family name indicates how many unique devices belong to that family.

for extended periods causes frost formation on its surface, which prematurely halts its operation due to short circuits. To overcome this, we first reduce chamber humidity to below 25% at room temperature with the DUT placed inside an airtight package and then seal the package before cooling the DUT.

#### D. Statistical Tests

| NIST Statistical Tests         |                           |  |  |  |  |
|--------------------------------|---------------------------|--|--|--|--|
| Monobit                        | Universal Statistical     |  |  |  |  |
| Frequency Within a Block       | Linear Complexity         |  |  |  |  |
| Runs                           | Serial                    |  |  |  |  |
| Longest-Run-of-Ones in a Block | Approximate Entropy       |  |  |  |  |
| Binary Matrix Rank             | Cumulative Sums           |  |  |  |  |
| Discrete Fourier Transform     | Random Excursions         |  |  |  |  |
| Non-Overlapping Template       | Random Excursions Variant |  |  |  |  |
| Overlapping Template Matching  |                           |  |  |  |  |

TABLE II: List of tests in the NIST Statistical Test Suite

No single test fully quantifies randomness, as apparent nondeterminism may stem from limited understanding, while truly random processes can occasionally show low entropy due to pure chance. Over years of deliberation, this challenge has led to the development of statistical test suites that assess randomness from multiple analytical perspectives. The National Institute of Standards and Technology (NIST) provides a widely-adopted test suite [69], consisting of 15 different tests (Table II) to evaluate the statistical quality and unpredictability of bit sequences. Industry standards require RNGs to pass NIST certification by analyzing long continuous sequences that rule out short-term entropy fluctuations. Following suit, the initial step in our framework is to verify RNG robustness in the continuous collection mode. However, passing the NIST tests in this manner is insufficient to claim security, as their diagnostic reliability depends strongly on how the data is collected, segmented, and concatenated before analysis.

#### E. Weakness Discovery Tests

To characterize TRNG behavior across environmental corners, we supplement NIST tests with additional tests aimed at exploring characteristics that reveal deeper insights into distribution quality, entropy behavior, and spatial correlation.

1) Collision Tests: Consider an RNG that outputs n uniformly distributed random bits at a time. How many such n-bit random sequences do we need before two sequences collide? For a total of  $2^n$  different possible combinations,  $2^n + 1$  sequences guarantee that at least one sequence repeats, an intuitive fact known as the pigeonhole principle. However, it is more likely than not (> 0.5 probability) that a collision occurs at a much smaller number of samples than intuition dictates, known as the birthday paradox.

The minimum number of samples required (k) for the probability (p) that at least one collision occurs is given by:

$$k \simeq \sqrt{2 \cdot 2^n \cdot \ln(\frac{1}{1-p})} \tag{1}$$

The birthday problem [53] details how the probability of collision becomes > 0.5 at a sample size of 302 and > 0.99 (guaranteed collision for practical purposes) at 777 samples, a number much lower than the intuitive 65,537 samples. Interestingly, when the generated sequence exhibits a non-uniform distribution—indicating a reduction in RNG output unpredictability and hence, its overall security—the practical sample size required to observe a collision diverges from the theoretically expected value based on uniform randomness. To better quantify this, we consider the equation of expected number of collisions (E) for a given number of samples collected (k):

$$E = \binom{k}{2} \cdot (\frac{1}{2^n}) \tag{2}$$

Our primary goal in this experiment is to check whether environmental stresses cause deviations in the observed number of collisions from the expected value. Moreover, we want to find out whether recognizable patterns emerge in these deviations under different environmental conditions.

2) Entropy: Another way to think of "non-uniformness" in RNG generated distributions is entropy. Entropy is the measure of uncertainty in an RNG's output or in the context of a TRNG, the measure of operational noise in the output. Various entropy estimation metrics exist, the two most popular ones being min-entropy and Shannon's Entropy, belonging to the Renyi family of entropies [17]. min-entropy measures the worst-case uncertainty of a random sequence. It encapsulates the odds

of an attacker with a priori knowledge of the RNG's outputs at guessing the next output. Naturally, for a knowledgeable attacker, the best strategy is to guess the most likely output:

$$min\text{-}entropy = -\log_2(\max_i p(x_i))$$
 (3)

Shannon's Entropy on the other hand, measures the average uncertainty of a random sequence. It encapsulates the guessing odds of an attacker with no prior knowledge of the RNG's outputs. The best strategy in this case is blind guessing, captured in Shannon's Entropy calculation:

Shannon's 
$$Entropy = -\sum_{i=1}^{m} p(x_i) \log_2 p(x_i)$$
 (4)

where m is the total number of collected samples. For a perfectly random generator with  $2^n$  possible n-bit outcomes, all outcomes occur with equal probability. Consequently, both Shannon's Entropy and min-entropy attain the theoretical maximum of n, meaning that each bit contributes equally to the total randomness of the sequence, both on average and in the worst-case sense. The goal of this experiment is to quantify the drop in operational noise of RNGs as we vary the environmental conditions.

3) Moran's I: Moran's I is a statistical measure of spatial autocorrelation that quantifies the degree of clustering in spatial data. We apply it to bitmaps to identify and compare the clustering of data. It is defined as:

Moran's 
$$I = \frac{N}{W} \cdot \frac{\sum_{i=1}^{N} \sum_{j=1}^{N} w_{ij} (x_i - \bar{x})(x_j - \bar{x})}{\sum_{i=1}^{N} (x_i - \bar{x})^2}$$
 (5)

where N is the number of elements in the bitmap,  $w_{ij}$  denotes the spatial weight between elements i and j (we calculate this using shared borders and k-nearest neighbors),  $W = \sum_{i=1}^{N} \sum_{j=1}^{N} w_{ij}$ ,  $\bar{x}$  is the mean value of the dataset, and  $x_i$  is the value of element i. Moran's I is bounded in [-1, 1], with -1 indicating complete dispersion, 0 indicating spatial randomness, and 1 signifying perfect clustering. For a truly random bitmap, the expected Moran's I value is 0, representing the absence of spatial correlation.

