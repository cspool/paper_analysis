# *D. Pseudo Random Number Generators (PRNGs)*

PRNGs use deterministic algorithms (e.g., cryptographic hashes or stream ciphers) to expand a small, high-quality seed into a long random-looking sequence. Their security depends entirely on the secrecy and unpredictability of that seed. In practice, TRNGs and PRNGs are combined: the PRNG "stretches" a small amount of randomness from the TRNG to create many keys, such as in Linux's /dev/urandom.

## III. THREAT MODEL

Ultra-low Size, Weight, and Power (UlSWaP) devices are deeply embedded into society, granting an adversary physical access. This proximity, combined with the known sensitivity of hardware True Random Number Generators (TRNGs) to operational and aging attacks [11], [32], [49], [51], [58], [79], [98], forms the basis of our threat model.

Traditional TRNG threat models fail to capture cross-layer effects because they isolate attacks into separate categories (e.g., operational, physical, or software). Since intermittent systems inherently entail a close coupling of hardware and software, we propose a unified, cross-layer threat model. This model aligns with contemporary research on these emerging devices [29], [49], [63], [70] and serves as the foundation for evaluating the robustness of their TRNGs.

Adversary Capabilities: Our adversary is a Level 2 physical attacker following the FIPS 140 taxonomy [59], where the microcontroller package acts as the security perimeter. The attacker is capable of invasive, non-destructive manipulation and possesses the following key capabilities:

- Power and Environmental Control: The attacker can manipulate the device's power supply (e.g., through brownouts or surges) and remove off-chip components to bypass regulation circuitry. They also have control over ambient environmental conditions, including temperature, humidity, and supply voltage. This control enables them to control power cycles and accelerate the aging process of the internal transistors non-destructively.
- Physical Monitoring: The attacker can probe package pins to monitor board-level communication, e.g., traffic between external TRNGs, memory chips, and the MCU.
- Software Tampering: The attacker can leverage standard programming interfaces, such as debuggers and Direct Memory Access (DMA), to inspect or tamper with the victim's software state and memory. This includes the ability to load untrusted software.

Following the threat model of related operational attacks [48], we consider devices that are permanently locked against programming and software updates to be outside the scope of our threat model. This work also does not focus on side-channel attacks, as our goal is not to exfiltrate the key material but to assess the quality of its source. This unified model serves as the basis for evaluating TRNG robustness under physically accessible, environmentally controlled conditions typical of intermittent computing systems.

