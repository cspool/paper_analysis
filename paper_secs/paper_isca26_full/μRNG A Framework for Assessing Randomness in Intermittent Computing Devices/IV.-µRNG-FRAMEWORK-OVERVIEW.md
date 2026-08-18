# IV. µRNG FRAMEWORK OVERVIEW

The goal of a True Random Number Generator (TRNG) is to produce an unpredictable sequence of bits to be used for ephemeral keys and nonces. This means that seeing the entire history of bits produced by a TRNG does not help you guess the next bit (or set of bits) that it produces (i.e., bits are independent). Probabilistically, the occurrence for each possible number that an ideal TRNG generates is equal, i.e., *uniformly distributed*. Given the sensitivity of hardware-based

![](_page_3_Figure_0.jpeg)

Fig. 2:  $\mu$ RNG testing framework.

noise sources to their environment, it is possible to influence this probability of occurrence.  $\mu$ RNG testing framework combines different experiments to quantify and influence the quality of a TRNG's output. Based on the outcome of these experiments, we classify RNGs into classes:

- Insecure: do not use in any scenario.
- Class 1: suitable for nominal, continuous operation.
- Class 2: suitable for nominal, intermittent operation.
- Class 3: suitable for extreme, intermittent operation as long as all software is trusted.
- Class 4: suitable for extreme, intermittent operation, possibly weak.
- Class 5: suitable for extreme, intermittent operation, recommended.

Figure 2 presents the experimental stages of  $\mu$ RNG. The framework evaluates and characterizes RNG robustness under varying operating conditions. Before every experiment, we uncover the ground truth configuration of the RNG. Our goal is to capture the RNG output before any hardware or software modifies it. This ensures that we characterize and evaluate the raw TRNG sequence without masking any effects of drop in entropy. We begin our experiments by collecting a continuous batch of outputs under nominal conditions and analyze them statistically to quantify their quality. Many well-established test suites exist for statistical analysis, such as NIST [69], Diehard [16], [52] or TestU01 [44]. If the RNG passes this stage, we proceed to collect outputs under intermittent operation and varying environmental conditions to assess RNG strength in deeply-embedded intermittent settings.

For the second analysis stage we introduce the effects of intermittency on the RNG. Intermittent systems are plagued by common-case restarts; this is often when RNGs are least secure as they have not accrued enough entropy [28], [30]. Thus, key material and nonces will always come shortly after

power-on in intermittent computing systems. To replicate the effects of intermittency on key generation, we take the first set of bits from the RNG after each power cycle of the target device. We then stitch these initial RNG outputs together to form a longer bit stream. Just as in the previous stage, we use existing tools to analyze the quality of the resulting bit stream.<sup>2</sup> Once the key material has been recorded, we power cycle the device.

For the third stage we determine if the RNG is actually a Pseudo-RNG (PRNG) or a TRNG. If it is a PRNG, then environmental variation has no impact on the resulting bit stream because its behavior is entirely software defined. In this case, we analyze the security of the seed used by the PRNG from a co-resident software attacker. This analysis centers on uncovering how the device reads, updates, and protects the key from being accessed by unprivileged software. In the case of a TRNG, we analyze the security of the TRNG with respect to a physical attacker.

The headless nature of UISWaP devices means TRNGs must harvest randomness from analog domain noise; this makes them sensitive to environmental conditions like temperature and voltage. The fourth stage of analysis accounts for effects of the environment on TRNG output: for each device, we define extreme voltage/temperature corners of operation. For each corner of operation, we perform the same data collection and analysis that we did for the second analysis stage.

Devices that still produce bit streams that pass all statistical tests move to the fifth and final analysis stage: weakness detection. In this stage, we couple existing statistical tests with additional low-level tests to determine whether there is a correlation between environmental conditions and the TRNG output quality. Note that in this case, quality is no longer binary as we are looking for a weakness that a determined/powerful attacker might be able to exploit in the future.<sup>3</sup>

<sup>&</sup>lt;sup>1</sup>TRNG post-processing only serves to mask inherent TRNG weakness deterministically. Thus, we must analyze the raw TRNG outputs to assess its true strength.

<sup>&</sup>lt;sup>2</sup>These test suites evaluate randomness primarily by detecting statistical deviations in a bitstream, agnostic of its semantic "continuity".

<sup>&</sup>lt;sup>3</sup>As with DES, MD5, and SHA1, weakness often precedes full compromise.

![](_page_4_Picture_0.jpeg)

Fig. 3: Power controller circuit diagram.

