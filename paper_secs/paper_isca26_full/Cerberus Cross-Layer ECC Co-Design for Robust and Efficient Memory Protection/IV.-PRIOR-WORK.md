# IV. PRIOR WORK

The inefficiency and limited reliability of multi-layer memory protection in commercial DRAMs have inspired numerous academic efforts to redesign ECC architectures. Prior work largely follows two directions: (i) schemes that consolidate protection into a strengthened S-ECC, and (ii) schemes that explicitly coordinate layers and leverage cross-layer information. This section summarizes representative research along these lines.

## *A. Stronger System ECC*

- *1) DUO:* DUO [68] bypasses on-die ECC and repurposes its internal redundancy at the system level. By forwarding redundant bits to the host through additional transfer beats, DUO extends S-ECC into a longer symbol-based codeword (e.g., RS(76,64)), providing SDDC protection. A portion of this redundancy is reserved as *on-chip redundancy parity*, which supports burst-erasure correction and verification, enabling recovery even when a complete chip failure coincides with multiple hard defects (Fig. 3a).
- *2) Unity ECC:* Unity ECC [22] presents a single-layer S-ECC framework capable of correcting both double-bit errors and single-chip failures. By allowing S-ECC to handle both frequent bit faults and rare device-level errors, Unity ECC

eliminates the need for on-die correction. This simplification reduces latency, power, and area overheads associated with O-ECC while maintaining robust end-to-end protection (Fig. 3b).

*3) Dual-Axis ECC:* Dual-Axis ECC [69] protects both storage and transfer errors via two-orientation decoding: vertical symbols correct storage faults, while horizontal overlays correct common DQS-induced transfer errors by reusing spare syndromes, eliminating L-ECC overhead and reducing retransmissions without extra redundancy.

## *B. Cross-layer Collaboration*

- *1) XED:* XED [65] proposes cooperative interaction between S-ECC and O-ECC to achieve SDDC protection. It exposes on-die ECC detection outcomes to the system through a predefined *catch-word* interface, enabling S-ECC to perform erasure-based decoding with improved correction efficiency in configurations with multiple DRAM devices per channel.
- *2) HARP:* HARP [70] improves reliability through profiling-based repair under O-ECC. It leverages O-ECC's information (e.g., via decode-bypass and/or correction-event reporting) to profile vulnerable locations and then applies repair actions to mitigate future errors at those locations. Its benefit depends on profiling coverage and available repair resources.

Table I summarizes industry practice and two representative academic designs (DUO and Unity ECC) for single-deviceper-channel organizations. Collectively, these academic approaches demonstrate the potential of single-layer ECC to improve system reliability and reduce redundancy. However, they overlook a critical industry requirement—*error concealment within DRAM devices*. To preserve product quality perception, DRAM vendors deliberately mask internal fault behavior, reporting only a limited set of vulnerable regions rather than exposing raw error counts [71]. Although academic proposals enhance transparency and end-to-end reliability, their exposure of device-level errors to the host conflicts with this industrial practice, raising concerns over warranty obligations and vendor accountability.

## V. CERBERUS

This section introduces Cerberus, a unified, *cross-layer* ECC framework that delivers high reliability at low overhead with robust end-to-end protection. Unlike conventional multilayer schemes in which each layer is designed and operated independently, Cerberus *co-designs* the generator and paritycheck matrices across layers to ensure interoperability. This interoperability allows a single encoder to produce redundancy shared across layers—each serving a distinct purpose (link protection, error concealment, or ultra-high reliability)—while preventing destructive cross-layer interference (e.g., avoiding miscorrections that amplify errors for the next layer). The following sections present the Cerberus architecture and its operational flow.

## *A. Architecture*

Cerberus targets single-device memory channels in which the channel transfer unit matches the device's internal access unit (e.g., HBM, LPDDR). In this paper, we focus on an HBM configuration with a 256-bit (32B) access unit and a total redundancy budget of 12.5% (4B per 32B data block). The 32B access unit aligns with current HBM practice, whereas the redundancy budget is lower than the HBM baseline (18.8%). Despite using less redundancy, Cerberus delivers stronger protection, making it suitable for custom HBMs and future HBM generations.

Fig. 4 illustrates the overall architecture. The framework consists of a single shared encoder ( 1 ) and three layerspecific decoders ( 1 , 2 , 3 ) that cooperate along the data path. The 32-bit redundancy generated once by the encoder is reused—wholly or partially—by the following decoding layers: (1) the *Link Layer* (LL), which provides early detection of write-path link errors; (2) the *Device Layer* (DL), which performs on-die error correction and concealment; and (3) the *System Layer* (SL), which ensures strong end-to-end, system-level reliability. Each layer interprets the shared 32 bit redundancy according to its role. The LL utilizes 16 bits exclusively for error detection; the DL reuses that same 16-bit portion for bit-level single-error correction (SEC) within the die; and the SL leverages the full 32 bits to perform symbolbased correction and detection. This *Encode-Once, Decode-Many* (EODM) organization with hierarchical redundancy reuse eliminates repeated encoding stages, reduces latency, storage overheads, and preserves seamless protection coverage across all layers without reliability gaps.

## *B. Cerberus Operations*

*1) Write Operation:* On a write, the encoder ( 1 ) takes 256-bit user data D and appends 32 bits of redundancy composed of R<sup>1</sup> and R<sup>2</sup> (16 bits each). This redundancy is produced by a generator matrix G<sup>S</sup>-ECC, which is a product of two sub-matrices, G<sup>1</sup> and G<sup>2</sup> (G<sup>S</sup>-ECC = G<sup>1</sup> · G2). G<sup>1</sup> first maps 256-bit D to a 272-bit intermediate codeword (D+R1); G<sup>2</sup> then maps this 272-bit word to a 288-bit final codeword ((D+R1)+R2). Although the description separates these as two encoding stages for conceptual clarity, practical

![](_page_6_Figure_7.jpeg)

Fig. 4: An overview of Cerberus

implementations can perform both mappings in a single step using the composite matrix GS-ECC.

The host omits conventional write-path L-ECC and transmits the 288-bit codeword ((D+R1)+R2) directly to DRAM. On the DRAM side, the first decoder ( 1 ) replaces L-ECC by verifying the link integrity of ((D+R1)+R2) using R<sup>2</sup> via the parity-check matrix H<sup>2</sup> (the dual of G2). Upon a mismatch, the decoder does not attempt correction but instead requests retransmission (e.g., via a conventional ALERT signal). The 16 bit redundancy provides early detection of transfer errors with a detection capability comparable to CRC16. If verification succeeds, the validated codeword ((D+R1)+R2) is stored in DRAM with its redundancy preserved for subsequent deviceand system-level reuse.

*2) Read Operation:* On a read, the second decoder ( 2 ) operates inside the bank group to perform *on-die* error correction and concealment. The device validates ((D+R1)+R2) using the same parity-check matrix H2, but this time, it applies the correction (SEC) when needed. This ensures that singlebit errors are corrected within the device, preventing their propagation to the memory controller.

Importantly, H<sup>2</sup> is designed to satisfy a *bounded-fault* constraint at the 16-bit symbol granularity: any miscorrection that arises from multiple flipped bits is *confined* to the originally faulty 16-bit symbol and cannot corrupt additional symbols. Consequently, device-level actions never increase symbol-level error severity. After local correction, the DRAM bypasses read-path L-ECC generation and instead forwards the corrected 288-bit codeword ((D+R1)+R2) to the controller.

At the controller, the third decoder ( 3 ) provides systemlevel protection using both redundancy fields (R1, R2). Leveraging the full 32-bit redundancy, it delivers SSC+DEC capability: it corrects either a single 16-bit symbol error (e.g., from a device-level miscorrection or critical peripheral-circuit faults) or two bit errors located in distinct symbols (e.g., overlapping storage/transfer errors). For more severe patterns, it still provides safe detection with probability 99.97%. It also extends correction beyond bank-group boundaries and replaces read-side L-ECC. When the decoder detects an uncorrectable error, it issues a single retry. If the retry is still uncorrectable, it reports a DUE. Otherwise, it treats the first event as a transient read-link or peripheral error and forwards the retry result, as in conventional L-ECC retry.

Collectively, Cerberus delivers SSC+DEC correction to the system, SEC correction to DRAM devices, and CRC16-level detection for link protection, meeting the distinct requirements of DRAM reliability and manufacturability. Compared to typical HBM designs, Cerberus offers stronger end-to-end correction at 12.5% overhead (vs. 18.8%) and is not confined to a single bank group. Compared to LPDDR-style protection, it matches the 12.5% overhead but upgrades from bit-level SEC to symbol-aware SSC+DEC at the host, while retaining on-die SEC and link detection for full end-to-end coverage.

## *C. Cross-Layer ECC Collaboration*

This architecture raises a key question: how can the system and device layers collaborate so that redundancy provisioned for the system layer is reusable in the device layer? We address this question in the following subsections by describing our device/link-layer, system-layer, and cross-layer ECC designs.

*1) Device/Link Layer:* We begin by describing the decoding mechanism at the device and link layers. These layers receive 288-bit data ((D+R1)+R2) and use R<sup>2</sup> to detect and correct errors in D and R1, resulting in a parity-check matrix H<sup>2</sup> with dimensions 16 × 288.

The device layer targets SEC-DED with a bounded-fault constraint at 16-bit granularity. We choose SEC based on observations from HBM3/4, where strong on-die correction can trigger severe miscorrections and still leave the system exposed to faults outside the bank group. Such miscorrections then force the system layer to provision additional correction strength and redundancy. For example, an O-ECC using 8-bit SSC with 2-symbol redundancy can miscorrect a two-symbol error into a three-symbol corruption, which would require 6 symbol redundancy in S-ECC for full recovery. Moreover, because error sources are not confined to a single bank group, strong intra-bank protection is helpful but offers limited reliability beyond the bank group. Therefore, we design the device layer to handle frequent small-scale bit errors, while the system layer is responsible for rarer but more severe faults.

To achieve SEC-DED with bounded-fault recovery, H<sup>2</sup> must satisfy the following conditions:

- Every column must be non-zero.
- SEC: All columns must be unique.
- DED: The sum of any two distinct columns must not be equal to any other column.
- Bounded fault: The sum of columns within any 16 column region must not match any column belonging to another region.

Additionally, since H<sup>2</sup> is also used in the link layer, it must be able to detect frequent transfer errors. To achieve this, we design H<sup>2</sup> to meet both the device-layer constraints and the CRC8 requirements. Unlike CRC16, CRC8 does not guarantee detection of all burst errors of length 9–16; however, the combined 16-bit redundancy still provides a random-error detection probability of 1 − 2 <sup>−</sup><sup>16</sup> (≈99.998%). Detecting 16 bit burst errors is especially important in DDR systems, where the burst length is typically 16, whereas in memory systems with shorter bursts (e.g., HBM4 with burst-8), this level of detection is less critical. To realize CRC8, H<sup>2</sup> must satisfy the condition that any eight consecutive columns are linearly independent.

As a result, we use H<sup>2</sup> to protect both the device and link layers, implementing SEC-DED with a bounded-fault property while also meeting CRC8 and providing CRC16-level detection capability.

*2) System Layer:* The system layer is responsible for ensuring end-to-end reliability by detecting and correcting errors that may have propagated through the device and link layers. This layer is designed to provide robust protection against severe, rare errors that escape the correction capabilities of the device and link layers. The system layer achieves this by leveraging the redundancy provided by the device layer (via R<sup>1</sup> and R2) to perform symbol-based error correction, leading to a parity-check matrix (HS-ECC) with dimensions of 32 × 288.

To achieve this, the system layer is designed to correct single 16-bit symbol errors or double-bit errors (SSC+DEC). This dual capability addresses both clustered errors caused by malfunctioning peripheral circuits (e.g., subwordline driver failures) and frequent, random bit errors that may occur simultaneously. The goal of the system layer is to maintain high levels of protection with minimal additional redundancy overhead, thereby ensuring end-to-end reliability without significant performance or storage costs.

To achieve SSC+DEC, HS-ECC must satisfy the following conditions:

- Every column must be non-zero.
- SSC: The sums of all symbol-aligned columns are unique.
- DEC: The sums of any two columns are unique.
- SSC+DEC: All sums from properties 2 and 3 should be unique (apart from double-bit errors in the same symbol, which are considered symbol errors).

For effective collaboration between the device/link layers and the system layer, HS-ECC must satisfy a single condition, which we discuss in the next section.

*3) Cross Layer:* This section describes how HS-ECC can be decomposed into H<sup>1</sup> and H<sup>2</sup> while encoding only once, and presents the single condition under which any 256-bit granularity DRAM can adopt the Cerberus framework. We begin by describing the shared-encoder generator matrix, G<sup>S</sup>-ECC.

We design G<sup>S</sup>-ECC to satisfy G<sup>S</sup>-ECC = G<sup>1</sup> · G2. For such a G<sup>1</sup> to exist, each row of G<sup>S</sup>-ECC must be expressible as a linear combination of the rows of G2. In other words, the row space of G<sup>S</sup>-ECC must be contained within the row space of G<sup>2</sup> (i.e., row(G<sup>S</sup>-ECC) ⊆ row(G2)). To ensure this condition in the parity-check matrix domain, we utilize the relationship between the generator matrix G and the parity-check matrix H (e.g., H2G<sup>⊤</sup> <sup>2</sup> = 0). In this domain, the condition is equivalent to the requirement that each row of H<sup>2</sup> be expressible as a linear combination of the rows of H<sup>S</sup>-ECC (i.e., row(H2) ⊆ row(H<sup>S</sup>-ECC)). If this simple condition holds, there are no additional constraints on adopting the Cerberus framework, which allows it to support a wide range of vendor-specific S-ECC schemes with high scalability.

![](_page_8_Figure_0.jpeg)

Fig. 5: The parity-check matrices of Cerberus for cross-layer design

## *D. Code Construction*

We derive an H-matrix that satisfies each layer's conditions through a two-step construction. First, we construct H<sup>2</sup> to provide SEC-DED with bounded-faults. We realize SEC-DED by assigning odd-weight columns. Instead of using a prefix region for bounded-faults, we enforce both the bounded fault and CRC8 properties by building the second half (8 columns) of each 16-column bounded region as XOR combinations of the columns in the first half (8 columns). This structure makes it easy to satisfy the CRC8 condition and also helps meet the SSC requirement of HS-ECC. Next, we map each bounded region of the binary H<sup>2</sup> to elements in GF(216) and place this symbolized H<sup>2</sup> (Fig. 5a) directly in the upper part of HS-ECC, thereby satisfying the cross-layer condition (row(H2) ⊆ row(HS-ECC)).

Second, we construct HS-ECC to satisfy both SSC and DEC. Since H<sup>2</sup> is already placed in the upper part of HS-ECC, we build the lower part using a greedy search. We randomly assign GF(216) elements to each symbol in the lower part, then binarize HS-ECC and check for syndrome overlaps to verify that the SSC and DEC conditions are met. If syndrome overlaps occur, we reconstruct the symbol with the largest number of overlaps and repeat this process until HS-ECC satisfies both SSC and DEC (Fig. 5b).

## *E. Hardware Implementation*

Cerberus largely reuses standard memory ECC primitives (e.g., an encoder and an SEC decoder) and updates only the G and H matrices (Fig. 6). On the write path, the controller encoder ( 1 ) computes R<sup>1</sup> and R<sup>2</sup> in a single pass using G<sup>S</sup>-ECC = G<sup>1</sup> · G2. It implements this multiplication with an XOR network (e.g., 8-level XOR trees). In DRAM, the first decoder ( 1 ) verifies writes by regenerating R<sup>2</sup> using a subset of the same XOR network.

On the read path, the second decoder in DRAM ( 2 ) generates a syndrome with an XOR-tree network and corrects single-bit errors based on the syndrome, which adds modest logic depth (≈ 4) to the combinational logic. It then forwards

![](_page_8_Figure_8.jpeg)

Fig. 6: The hardware implementation of Cerberus

the corrected 288-bit codeword (with redundancy retained) for end-to-end decoding. The third decoder in the controller ( 3 ) applies HS-ECC to the received 288-bit codeword to generate a 32-bit syndrome and runs SSC and DEC correctors in parallel [22]. The SSC corrector uses Chien search with a modified Berlekamp–Massey procedure [72], and the DEC corrector uses a block-pair solver [73].

Overall, the encoder and the first two decoders are on par with existing implementations. The main added complexity is the third decoder for SSC+DEC. However, it performs error *detection* within a single cycle to avoid latency increases on error-free accesses, and it completes correction within a single cycle by running SSC and DEC correction in parallel. Because errors are rare, we include only the detection latency in the performance evaluation (Section VI-B) and report the area overhead of advanced decoding (Section VI-C).

## VI. EVALUATION

We evaluate the reliability and performance impact of Cerberus and compare it with single-layer ECC and state-ofthe-art multi-layered DRAM ECC configurations.

TABLE II: A comparison of reliability against single-location error scenarios

|                   |                   |                    | Single-layer |             | Multi-layer        |                |                    |                    | Cross-layer    |                    |  |
|-------------------|-------------------|--------------------|--------------|-------------|--------------------|----------------|--------------------|--------------------|----------------|--------------------|--|
| Redundancy (%)    |                   |                    | 12.5% (32b)  | 18.8% (48b) | 12.5%              | (32b)          | 18.8%              | (48b)              | 12.5% (32b)    | 15.6% (40b)        |  |
| Error<br>Location | Error<br>Scenario | Decoding<br>Result | Unity ECC    | DUO         | LPDDR6/<br>SEC-DED | LPDDR6/<br>CRC | HBM4/<br>SEC-DED   | HBM4/<br>CRC       | Cerberus (32b) | Cerberus (40b)     |  |
|                   | SE (%)            | CE                 |              | 100.000     |                    |                |                    |                    |                |                    |  |
|                   |                   | CE                 | 100.000      |             | 0.048              | 0.024          | 100.000            |                    |                |                    |  |
|                   | 16E (%)           | DUE                | 0.000        |             | 99.563             | 99.976         | 0.000              |                    |                |                    |  |
|                   |                   | SDC                | 0.000        |             | 0.389              | 0.000          | 0.000              |                    |                |                    |  |
| In bank           |                   | CE                 | 0.003        | 1.558       | 0.003              | 0.000          | 0.003              | 0.003              | 0.003          | 0.003              |  |
| III bank          | 32E (%)           | DUE                | 99.972       | 98.407      | 99.585             | 99.999         | 99.997             | 99.997             | 99.971         | 99.997             |  |
|                   |                   | SDC                | 0.025        | 0.035       | 0.412              | 0.001          | $2 \times 10^{-5}$ | $1 \times 10^{-6}$ | 0.026          | $3 \times 10^{-4}$ |  |
|                   | SE+SE (%)         | CE                 | 100.000      |             | 10.515             | 0.000          | 4.988              | 4.940              | 100.000        |                    |  |
|                   |                   | DUE                | 0.0          |             | 89.485             | 100.000        | 94.980             | 95.060 0.000       |                |                    |  |
|                   |                   | SDC                | 0.0          | 000         | 0.000              | 0.000          | 0.032              | 0.000              | 0.0            | 000                |  |
|                   | SE (%)            | CE                 | 100.000      |             |                    |                |                    |                    |                |                    |  |
| Write             | DQE (%)           | CE                 |              | 100.000     |                    |                |                    |                    |                |                    |  |
| Link              | DQSE (%)          | CE                 | 0.000        | 0.000       | 99.998             | 99.998         | 49.994             | 49.991             |                | 999                |  |
| LIIIK             |                   | DUE                | 99.971       | 99.945      | 0.002              | 0.002          | 50.005             | 50.008             | 0.0            |                    |  |
|                   |                   | SDC                | 0.029        | 0.055       | 0.000              | 0.000          | 0.001              | 0.001              | 0.0            | 000                |  |
|                   | SE (%) CE         |                    | 100          |             | 100.000            | 0.000          | 100.000            | 0.000              |                | .000               |  |
|                   | 3E (70)           | DUE                | 0.0          |             | 0.000              | 100.000        | 0.000              | 100.000            |                | 000                |  |
|                   | DE (%)            | DE (%) CE          |              | 100.000     |                    | 0.000          |                    |                    |                | 100.000            |  |
| Out bank          |                   | DUE                | 0.000        |             |                    |                |                    | 0.0                | 0.000          |                    |  |
| (Read peri/link)  | DQE (%)           | CE                 | 100.000      |             |                    |                |                    |                    |                |                    |  |
| (redu periorine)  | DQL (70)          | DUE                |              |             |                    |                | 0.000              |                    |                |                    |  |
|                   |                   | CE                 | 0.000        | 0.000       | 99.998             | 99.998         | 49.994             | 49.991             | 99.972         | 100.000*           |  |
|                   | DQSE (%)          | DUE                | 99.971       | 99.945      | 0.002              | 0.002          | 50.005             | 50.008             | 0.000          | 0.000              |  |
|                   |                   | SDC                | 0.029        | 0.055       | 0.000              | 0.000          | 0.001              | 0.001              | 0.028          | $1 \times 10^{-4}$ |  |

<sup>\*</sup> Rounded to 100.000% for display; actual value is slightly lower (e.g., 99.9998%).

#### A. Error Coverage

We evaluate the reliability of Cerberus using Monte Carlo error-injection experiments. We inject random errors under various error scenarios, and quantify correction and detection through ECC decoding. Based on system-level analyses of DRAM faults [15]–[17], [58], [74], we derive location-specific error patterns, detailed below.

We consider three error locations: (i) *In bank*, (ii) *Write link*, and (iii) *Out bank*. First, *In bank* covers faults internal to a DRAM bank. We consider the following error scenarios: Single Error (SE; caused by cell or BLSA), 16-bit Error (16E; CSL or SWL), 32-bit Error (32E; SWD) and a combination of two Single Errors (SE+SE). Second, *Write link* denotes the transmit path during writes, including transmission-induced faults. We select error scenarios: SE, Data Pin Error (DQE), and Data Strobe Error (DQSE). Third, *Out bank* spans the read path beyond the bank—device periphery and I/O—and we select error scenarios: SE, Double Error (DE; peripheral errors (e.g., TSV)) and DQE, DQSE (read-link errors).

For each error scenario, we inject errors at the specified DRAM location and flip the designated number of bits, each independently with a 50% probability. We evaluate both single-location and multi-location cases. Each ECC scheme classifies outcomes as correctable (CE), detectable but uncorrectable (DUE), or undetectable (SDC), and we aggregate results over 10 million iterations. For link errors, we count DUEs as CEs due to retransmission and retry.

We compare Cerberus against a range of layered ECC configurations. For single-layer schemes, we use Unity ECC [22] and DUO [68], both deployed on single-device memory. For multi-layer schemes, we evaluate state-of-the-art DRAM configurations, LPDDR6 [25] and HBM4 [26], which employ S-ECC as either SEC-DED or CRC [37], [52]. Finally, we

evaluate the cross-layer framework Cerberus with both 12.5% (32b) and 15.6% (40b) redundancy.

1) Single-Location: Table II summarizes error coverage and redundancy for each ECC configuration under single-location scenarios. For the *In bank*, all configurations correct SE and 16E at 100% except for 16E in LPDDR6. Because LPDDR6 partitions the 12.5% redundancy per layer, it lacks sufficient budget to correct 16E. Moreover, LPDDR6/SEC-DED shows a high SDC rate for 16E and 32E due to O-ECC miscorrections. By contrast, Cerberus guarantees 100% correction of 16E while using the same 12.5% total redundancy budget across layers. For other scenarios, Cerberus provides strong detection capability for 32E and, with increased redundancy, can further enhance robustness.

For SE+SE, none of the multi-layer configurations provides guaranteed correction. In LPDDR6, the SEC-DED O-ECC detects the event but forwards the uncorrected data to the controller without a hint, while the downstream S-ECC (SEC-DED or CRC) lacks sufficient correction capability (except in cases where errors occur in on-die parity bits that are not transferred). Similarly, in HBM4, the SSC O-ECC corrects only when both errors fall within the same symbol; otherwise the residual pattern exceeds the S-ECC capability. This behavior aligns with prior observations on multi-layer protection [50], although the SDC rates differ because [50] considers SEC O-ECC (without double-error detection). Unity ECC and DUO correct SE+SE by concentrating redundancy in a strengthened single layer that supports double-bit correction. Cerberus also corrects SE+SE via its SSC+DEC while preserving on-die error concealment for the common case of single-bit errors.

For the *Write link*, Unity ECC and DUO can correct SE and DQE with strong S-ECC, but wider transfer errors (DQSE) are only *detected*, not corrected. Because they are single-layer

TABLE III: A comparison of reliability against multi-location error scenarios

|                   |                   |                    | Single-layer |             | Multi-layer        |                |                     |                    | Cross-layer    |                    |  |
|-------------------|-------------------|--------------------|--------------|-------------|--------------------|----------------|---------------------|--------------------|----------------|--------------------|--|
| Redundancy (%)    |                   |                    | 32b (12.5%)  | 48b (18.8%) | 32b (12.5%)        |                | 48b (18.8%)         |                    | 32b (12.5%)    | 40b (15.6%)        |  |
| Error<br>Location | Error<br>Scenario | Decoding<br>Result | Unity ECC    | DUO         | LPDDR6/<br>SEC-DED | LPDDR6/<br>CRC | HBM4/<br>SEC-DED    | HBM4/<br>CRC       | Cerberus (32b) | Cerberus (40b)     |  |
|                   | SE+               | CE                 | 100.000      |             | 100.000            | 0.000          | 100.000             | 0.000              | 100.000        |                    |  |
| In bank+          | SE (%)            | DUE                | 0.000        |             | 0.000              | 100.000        | 0.000               | 100.000            | 0.000          |                    |  |
| Out bank          | SE+               | CE                 | 0.947        | 100.000     |                    | 0.0            | 0.000 100.000       |                    |                |                    |  |
| Out bank          | DE (%)            | DUE                | 99.019       | 0.000       |                    | 100            | 0.000               |                    |                | 000                |  |
|                   | DE (%)            | SDC                | 0.034        | 0.000       |                    | 0.0            | 0.000               |                    |                | 000                |  |
|                   | 16E+<br>DQE (%)   | CE                 | 5.545        | 100.000     | 0.047              | 0.025          |                     |                    | 100.000        |                    |  |
|                   |                   | DUE                | 94.431       | 0.000       | 99.565             | 99.975         | 0.000               |                    |                |                    |  |
| In bank+          |                   | SDC                | 0.024        | 0.000       | 0.388              | 0.000          | 0.000               |                    |                |                    |  |
| Write Link        | 32E+<br>DQSE (%)  | CE                 | 0.000        | 0.000       | 0.002              | 0.000          | 0.001               | 0.001              | 0.003          | 0.003              |  |
|                   |                   | DUE                | 99.972       | 99.950      | 99.586             | 99.999         | 99.999              | 99.999             | 99.972         | 99.997             |  |
|                   |                   | SDC                | 0.028        | 0.050       | 0.412              | 0.001          | $2 \times 10^{-5}$  | $8 \times 10^{-7}$ | 0.025          | $3 \times 10^{-4}$ |  |
|                   | SE+               | CE                 | 1.293        | 100.000     | 100.000            | 0.000          | 100.000             | 0.000              | 100.000        |                    |  |
| In bank+          | SE+               | DUE                | 98.672       | 0.000       | 0.000              | 100.000        | 0.000 100.000 0.000 |                    | 000            |                    |  |
| Out bank+         | SE (%)            | SDC                | 0.035        | 0.000       | 0.000              | 0.000          | 0.000               | 0.000 0.000        |                | 000                |  |
| Write Link        | SE+               | CE                 | 0.000        | 0.000       | 99.998             | 99.998         | 50.010              | 49.982             | 99.998         | 99.999             |  |
| WITH LINK         | DQE+              | DUE                | 97.971       | 99.950      | 0.002              | 0.002          | 49.988              | 50.017             | 0.002          | 0.001              |  |
|                   | DQSE (%)          | SDC                | 0.028        | 0.050       | $3 \times 10^{-5}$ | 0.000          | 0.001               | 0.001              | 0.000          | 0.000              |  |

schemes, they cannot perform early detection, so corrupted data may be written back uncorrected, allowing subsequent faults to accumulate and increase the risk of severe reliability issues. In contrast, the multi-layer baselines employ L-ECC and thus offer high detection for SE/DQE/DQSE, enabling correction via retransmission. However, when the L-ECC is provisioned with limited redundancy, as in HBM4, DQSE detection drops to roughly half of cases. Cerberus, on the other hand, guarantees 100% detection (and thus correction) for SE and DQE, and for wider DQSE, the first decoder already provides high detection; any remaining cases are caught by the stronger third decoder, yielding a robust end-to-end design.

For the *Out bank*, Unity ECC and DUO can correct SE, DE, and DQE using a strong S-ECC. However, they only *detect* DQSE because they do not guarantee a retry after detection. In contrast, the multi-layer configurations do not guarantee correction for peripheral DE. Moreover, when they apply S-ECC with CRC (LPDDR6-CRC and HBM4-CRC), they fail to correct even SE and instead provide detection only. This limitation stems from redundancy partitioning across layers, which prevents redundancy reuse. For example, HBM4 can allocate 32b to a strong O-ECC (SSC) and leave only 16b for S-ECC. This allocation forces weaker codes (SEC-DED or CRC) and leaves the system vulnerable to out-of-bank errors. In contrast, Cerberus guarantees 100% correction for SE, DE, and DQE, and achieves near-complete correction for DQSE with retries.

2) Multi-Location: Table III summarizes the error coverage and redundancy of each ECC configuration under multi-location scenarios. For combined *In bank* and *Out bank* errors, Unity ECC and DUO guarantee 100% correction for SE+SE. However, for SE+DE, DUO still corrects while Unity ECC fails. Although Unity ECC corrects SE and DE individually in the single-location case, the overlap across two locations exposes the limitation of relying solely on S-ECC. The multi-layer approaches show the same behavior as in the single-location *Out bank* case (e.g., failing to correct peripheral DE), because although O-ECC corrects *In bank* SE, *Out bank* errors still remain due to the weaker code used for S-ECC.

In contrast, Cerberus guarantees correction for both SE+SE and SE+DE across locations even with a small redundancy budget, as O-ECC corrects the In bank SE and S-ECC covers all Out bank errors. For combined In bank and Write link, configurations with 12.5% redundancy (Unity ECC/LPDDR6) do not guarantee correction for 16E+DQE. However, Cerberus can correct it even with the same redundancy. For 32E+DQSE, Cerberus likewise provides significantly higher detection than other schemes with the same redundancy. Finally, in the highrisk scenario where errors occur simultaneously at all locations (In bank, Out bank, and Write link), single-layer configurations reveal the limitation of relying only on S-ECC, since all overlapping errors are exposed to the system layer. DUO can correct SE+SE+SE with its stronger scheme, but it fails once larger errors are involved, and Unity ECC also cannot guarantee correction across these cases. Multi-layer configurations improve locality by letting each layer handle its corresponding errors (e.g., O-ECC handles In bank errors and L-ECC handles link errors), but because the limited redundancy budget is divided across layers, they still fail to provide correction in all cases. In contrast, Cerberus retains the role of each layer while reusing redundancy across layers, enabling nearly 100% correction across all cases with a small redundancy budget and thus offering robust end-to-end reliability.

Rather than limiting our evaluation to in-DRAM faults, we assess reliability using error scenarios that span the entire memory system, and show that Cerberus maintains strong reliability under both single- and multi-location errors. In addition, Cerberus is a scalable framework that can accommodate higher redundancy. Although this increases overhead, it improves detection capability and yields a more robust system.

#### B. Performance & Energy Consumption

1) GPU Performance: We evaluate the performance impact of Cerberus (32b) and Cerberus (40b) on GPUs using the cycle-level simulator Accel-Sim [75], and compare it against HBM4 [26], Unity ECC [22], and DUO [68]. Our system model is based on an NVIDIA V100 GPU configured with 32 HBM channels, with detailed parameters listed in Table IV.

![](_page_11_Figure_0.jpeg)

Fig. 7: Comparison of GPU performance and DRAM energy for Cerberus across the evaluated benchmarks

TABLE IV: The simulation configuration

| Components | Configuration                                               |  |  |  |  |
|------------|-------------------------------------------------------------|--|--|--|--|
| # of SMs   | 80                                                          |  |  |  |  |
|            | 1132 MHz, 4 warp schedulers/SM,                             |  |  |  |  |
| SM         | up to 32 blocks/SM, up to 48 warps/SM                       |  |  |  |  |
| L1 cache   | Up to 128KiB, 4 banks, 128B line, 256-way, 384 MSHR entries |  |  |  |  |
| L2 cache   | 4MiB, 128B lines, 16-way, 192 MSHR entries                  |  |  |  |  |
| Memory     | 256B channel interleaving, 64-entry scheduling queue,       |  |  |  |  |
| controller | FR-FCFS scheduling                                          |  |  |  |  |
|            | HBM4, 32 channels, 6.4Gbps                                  |  |  |  |  |
| Memory     | tRCD = 30-cycle, tRRD = 4-cycle, CL = 24-cycle,             |  |  |  |  |
|            | tCCDS = 2-cycle, tCCDL = 4-cycle, WL = 14-cycle             |  |  |  |  |

To capture a wide range of application behaviors, we use 16 workloads drawn from four benchmark suites: Rodinia [76], Parboil [77], GraphBIG [78], and PolyBench [79].

The ECC decoder affects the time from a read command to the output of the first corresponding data beat (tCL), while the encoder impacts the write latency (tWL) [80]. We evaluate the performance by adjusting these two timing parameters for each ECC configuration. For HBM4, we estimate the encoder and decoder latency overheads of 16-bit SSC-based O-ECC as 2ns and 5ns, respectively [80]. In contrast, the latency overheads of both S-ECC (CRC) and L-ECC (parity) are less than 1ns for both encoding and decoding. For Unity ECC and DUO, we derive their timing parameters by first removing the O-ECC latency overhead from the HBM4 baseline and then adding each scheme's synthesized latency. Logic synthesis in a UMC 28nm process yields latency overheads of 1.46ns for Unity ECC and 1.92ns for DUO. For both Cerberus (32b) and Cerberus (40b), we remove the O-ECC encoder latency and reduce the decoder latency from 5ns to 2ns under the modified O-ECC scheme [80], since the two configurations have nearly identical O-ECC latency. We also model S-ECC latency from logic synthesis, adding 0.85ns for Cerberus (32b) and 0.89ns for Cerberus (40b). Finally, we convert all latency values into clock cycles at 1.6 GHz (6.4 Gbps after QDR in HBM4).

Fig. 7a reports the instructions per cycle (IPC) for HBM4, Unity ECC, DUO, Cerberus (32b) and Cerberus (40b), normalized to HBM4. Across benchmarks, Cerberus (32b) improves IPC by 0.2%, 0.2%, 1.1%, and 1.4% (0.7% geomean), while Cerberus (40b) improves IPC by 0.3%, 0.1%, 1.2%, and 0.4% (0.5% geomean). Unity ECC and DUO achieve similar IPC gains, but they provide lower reliability than Cerberus. In contrast, Cerberus delivers higher reliability than HBM4 while using less redundancy, yet still improves IPC. It also attains performance comparable to Unity ECC (0.9% geomean). These benefits stem from the EODM organization of Cerberus, which eliminates repeated encoding stages through efficient reuse of redundancy across layers.

*2) DRAM Energy Consumption:* Cerberus uses different storage and transfer bit widths than HBM4. To estimate the resulting DRAM power/energy, we use HBM2E operating currents from a datasheet [81].

We assume the precharge standby current (IDD2N) is independent of bit width, while the incremental activation current (IDD0−IDD2N) and active-standby current (IDD3N−IDD2N) scale with the stored bit count (e.g., (256+32+16) bits in HBM4 vs. (256+32) bits in Cerberus (32b)). For read/write activity, we partition the incremen-

TABLE V: Estimated DRAM operating currents (per pseudochannel)

| DRAM current        | Unity<br>ECC | DUO      | HBM4   | Cerberus<br>(32b) | Cerberus<br>(40b) |  |  |  |  |
|---------------------|--------------|----------|--------|-------------------|-------------------|--|--|--|--|
| IDD0 (mA)           | 47.42        | 47.56    | 47.56  | 47.42             | 47.49             |  |  |  |  |
| (ACT-PRE)           | (99.7%)      | (100%)   | (100%) | (99.7%)           | (99.9%)           |  |  |  |  |
| IDD2N (mA)          |              | 44.88    |        |                   |                   |  |  |  |  |
| (Precharge standby) | (100%)       |          |        |                   |                   |  |  |  |  |
| IDD3N (mA)          | 47.69        | 47.84    | 47.84  | 47.69             | 47.77             |  |  |  |  |
| (Active standby)    | (99.7%)      | (100%)   | (100%) | (99.7%)           | (99.8%)           |  |  |  |  |
| IDD4R (mA)          | 521.28       | 547.75   | 525.72 | 521.28            | 534.52            |  |  |  |  |
| (Read)              | (99.2%)      | (104.2%) | (100%) | (99.2%)           | (101.7%)          |  |  |  |  |
| IDD4W (mA)          | 365.27       | 383.07   | 368.25 | 365.29            | 374.17            |  |  |  |  |
| (Write)             | (99.2%)      | (104%)   | (100%) | (99.2%)           | (101.6%)          |  |  |  |  |

tal currents (IDD4R-IDD2N and IDD4W-IDD2N) between bank-group-internal transfer (cells $\rightarrow$ O-ECC) and bank-group-external transfer (O-ECC $\rightarrow$ processor) and apply a 61%:39% split from prior HBM2 analysis [8]. We then scale each component by the corresponding transfer width: HBM4 transfers (256+32+16) bits within a bank group and (256+16) bits outside, whereas Cerberus (32b) transfers (256+32) bits within a bank group and (256+32) bits within a bank group and (256+32) bits outside. Finally, we compute overall DRAM energy consumption using the Micron DDR4 power calculator [82] with these current values. Table V summarizes the DRAM operating currents used in our evaluation.

Fig. 7b shows the results. Cerberus (32b) reduces energy by 1.84% on average compared to HBM4. This is primarily because intra-die transfers from cells to bank peripherals often consume more energy than off-chip transfer [8], [83], and Cerberus reduces these bank-group-internal transfers. With higher redundancy, Cerberus (40b) consumes 0.86% more energy than HBM4 on average to provide stronger protection.

## C. Hardware Overheads

To estimate the hardware costs, we implement SystemVerilog models for the encoder and decoders. We synthesize these models with Synopsys Design Compiler using a UMC 28nm standard library. We then normalize the resulting area to NAND2 equivalents (the number of NAND2 gates that occupy the same area) to present process-independent results.

Table VI summarizes the area cost of Cerberus. Decoder 1 (L-ECC) and Decoder 2 (O-ECC) reside inside the DRAM device. Together, they require only 14,693 NAND2 equivalents in Cerberus (32b) and 15,662 NAND2 equivalents in Cerberus (40b), which correspond to 0.0074 mm² and 0.0079 mm², respectively. This overhead is negligible compared to an HBM stack footprint (e.g., 121 mm² for HBM3 [84]). Decoder 3 (S-ECC) dominates the overall overhead, mainly due to the DEC corrector. Even so, the processor-side overheads (encoder+S-ECC) total 127,583 NAND2 equivalents for Cerberus (32b) and 167,954 NAND2 equivalents for Cerberus (40b). Relative to modern GPUs with billions of transistors, this is a tiny fraction of the transistor count (e.g.,  $2.5 \times 10^{-6}$  to  $3.2 \times 10^{-6}$  of a 208B-transistor Blackwell [85]).

TABLE VI: Area overheads (in NAND2 equivalents)

|                      | Cerberus (32b)                      | Cerberus (40b)                      |
|----------------------|-------------------------------------|-------------------------------------|
| Encoder              | $1632.79 \ \mu \text{m}^2 \ (3240)$ | $2015.49 \ \mu \text{m}^2 \ (3999)$ |
| Decoder 1<br>(L-ECC) | 1205.40 $\mu \text{m}^2$ (2392)     | 1398.43 $\mu \text{m}^2$ (2775)     |
| Decoder 2<br>(O-ECC) | 6199.87 $\mu \text{m}^2$ (12301)    | 6495.15 $\mu \text{m}^2$ (12887)    |
| Decoder 3<br>(S-ECC) | 62669.04 $\mu \text{m}^2$ (124343)  | 82633.15 $\mu \text{m}^2$ (163955)  |

#### VII. CONCLUSION

This paper presents Cerberus, a cross-layer ECC co-design that addresses key challenges of multi-layer ECC: inefficient use of redundancy, overlapping protection coverage, and destructive cross-layer interference due to miscorrections. Implemented on HBM4 with a cross-layer ECC design, Cerberus reduces redundancy by 33.3% while still providing higher reliability through efficient redundancy reuse. Moreover, its Encode-Once, Decode-Many (EODM) architecture eliminates unnecessary encoding stages, improving performance and delivering seamless coverage without protection gaps. Overall, Cerberus provides a promising framework for achieving high reliability in future HBM- and LPDDR-based systems.

#### REFERENCES

- A. Spessot and H. Oh, "1T-1C Dynamic Random Access Memory Status, Challenges, and Prospects," *IEEE Transactions on Electron Devices*, vol. 67, no. 4, 2020.
- [2] S.-L. Gong, J. Kim, and M. Erez, "DRAM Scaling Error Evaluation Model Using Various Retention Time," in *Proceedings of the Annual IEEE/IFIP International Conference on Dependable Systems and Networks Workshops (DSN-W)*, 2017.
- [3] O. Mutlu, Main Memory Scaling: Challenges and Solution Directions. Springer New York, 2015.
- [4] H. Hassan, M. Patel, J. S. Kim, A. G. Yaglikci, N. Vijaykumar, N. M. Ghiasi, S. Ghose, and O. Mutlu, "CROW: A Low-Cost Substrate for Improving DRAM Performance, Energy Efficiency, and Reliability," in Proceedings of the 46th international symposium on computer architecture. 2019.
- [5] H. Ha, Understanding and Improving the Energy Efficiency of DRAM. Stanford University, 2018.
- [6] H. Park, S.-M. Yu, and J. Song, "An 11 Gb/s 0.376 pJ/Bit Capacitor-Less Dicode Transceiver With Pattern-Dependent Equalizations TIA Termination for Parallel DRAM Interfaces," *IEEE Access*, 2024.
- [7] Y. Jung, S. Lee, H. Kim, and S. Cho, "A Supply-Noise-Induced Jitter-Cancelling Clock Distribution Network for LPDDR5 Mobile DRAM featuring a 2nd-order Adaptive Filter," in *Proceedings of the International Solid State Circuits Conference (ISSCC)*, vol. 65, 2022.
- [8] M. O'Connor, N. Chatterjee, D. Lee, J. Wilson, A. Agrawal, S. W. Keckler, and W. J. Dally, "Fine-Grained DRAM: Energy-Efficient DRAM for Extreme Bandwidth Systems," in *Proceedings of the 50th Annual IEEE/ACM International Symposium on Microarchitecture*, 2017.
- [9] K. S. Yim, C. Pham, M. Saleheen, Z. Kalbarczyk, and R. Iyer, "Hauberk: Lightweight Silent Data Corruption Error Detector for GPGPU," in Proceedings of the International Symposium on Parallel and Distributed Processing (IPDPS), 2011.
- [10] D. Fiala, F. Mueller, C. Engelmann, R. Riesen, K. Ferreira, and R. Brightwell, "Detection and Correction of Silent Data Corruption for Large-Scale High-Performance Computing," in *Proceedings of the International Conference on High Performance Computing, Networking, Storage and Analysis (SC)*, 2012.
- [11] R. Yeleswarapu and A. K. Somani, "Addressing multiple bit/symbol errors in DRAM subsystem," *PeerJ Computer Science*, vol. 7, 2021.

- [12] M. B. Sullivan, M. T. I. Ziad, A. Jaleel, and S. W. Keckler, "Implicit Memory Tagging: No-Overhead Memory Safety Using Alias-Free Tagged ECC," in *Proceedings of the International Symposium on Computer Architecture (ISCA)*, 2023.
- [13] S. Li, D. H. Yoon, K. Chen, J. Zhao, J. H. Ahn, J. B. Brockman, Y. Xie, and N. P. Jouppi, "MAGE: Adaptive Granularity and ECC for Resilient and Power Efficient Memory Systems," in *Proceedings of the International Conference on High Performance Computing, Networking, Storage and Analysis*, 2012.
- [14] B. Salami, O. S. Unsal, and A. C. Kestelman, "Evaluating Built-In ECC of FPGA On-Chip Memories for the Mitigation of Undervolting Faults," in *Proceedings of the 27th Euromicro International Conference on Parallel, Distributed and Network-Based Processing (PDP)*, 2019.
- [15] M. V. Beigi, Y. Cao, S. Gurumurthi, C. Recchia, A. Walton, and V. Sridharan, "A Systematic Study of DDR4 DRAM Faults in the Field," in *Proceedings of the International Symposium on High Performance Computer Architecture (HPCA)*, 2023.
- [16] H. Chung, E. Oh, S. Baek, H. Yoon, J. Yoo, S. Lee, Y. Lee, A. Bramhanand, B. Dodds, Y. Zhou, and N. S. Kim, "DRAM Fault Classification through Large-Scale Field Monitoring for Robust Memory RAS Management," in *Proceedings of the International Symposium on Microarchitecture (MICRO)*, 2025.
- [17] J. Jung and M. Erez, "Predicting Future-System Reliability with a Component-Level DRAM Fault Model," in *Proceedings of the 56th Annual IEEE/ACM International Symposium on Microarchitecture*, 2023.
- [18] A. Bacchini, M. Rovatti, G. Furano, and M. Ottavi, "Characterization of Data Retention Faults in DRAM Devices," in *Proceedings of the International Symposium on Defect and Fault Tolerance in VLSI Systems (DFT)*, 2014.
- [19] Y. Moon, S. Shin, S. Jang, D. Won, and S. Kang, "A Novel Prediction-Based Two-Tiered ECC for Mitigating SWD Errors in HBM," *IEEE Transactions on Very Large Scale Integration (VLSI) Systems*, 2024.
- [20] M. Patel, J. S. Kim, H. Hassan, and O. Mutlu, "Understanding and Modeling On-Die Error Correction in Modern DRAM: An Experimental Study using Real Devices," in *Proceedings of the International Conference on Dependable Systems and Networks (DSN)*, 2019.
- [21] M. Patel, J. S. Kim, T. Shahroodi, H. Hassan, and O. Mutlu, "Bit-Exact ECC Recovery (BEER): Determining DRAM On-Die ECC Functions by Exploiting DRAM Data Retention Characteristics," in *Proceedings of the International Symposium on Microarchitecture (MICRO)*, 2020.
- [22] D. Kim, J. Lee, W. Jung, M. B. Sullivan, and J. Kim, "Unity ECC: Unified Memory Protection Against Bit and Chip Errors," in *Proceedings of the International Conference on High Performance Computing, Networking, Storage and Analysis (SC)*, 2023.
- [23] R. Xie, A. U. Haq, Y. Fang, L. Ma, S. Sen, S. Venkataramani, L. Liu, and T. Zhang, "Breaking the HBM Bit Cost Barrier: Domain-Specific ECC for AI Inference Infrastructure," *IEEE Computer Architecture Letters*, 2025.
- [24] JEDEC standard, "Double Data Rate (DDR) 5," in *JESD79-5C.01*, 2024.
- [25] ——, "LPDDR6 standard," in *JESD209-6*, 2024.
- [26] ——, "High Bandwidth Memory (HBM4) DRAM," in *JESD270-4*, 2025.
- [27] D. H. Yoon and M. Erez, "Virtualized and Flexible ECC for Main Memory," in *Proceedings of the fifteenth International Conference on Architectural support for programming languages and operating systems*, 2010.
- [28] K. Criss, K. Bains, R. Agarwal, T. Bennett, T. Grunzke, J. K. Kim, H. Chung, and M. Jung, "Improving Memory Reliability by Bounding DRAM Faults: DDR5 improved reliability features," in *Proceedings of the International Symposium on Memory Systems (MEMSYS)*, 2020.
- [29] L. Cojocar, K. Razavi, C. Giuffrida, and H. Bos, "Exploiting Correcting Codes: On the Effectiveness of ECC Memory Against Rowhammer Attacks," in *Proceedings of the IEEE Symposium on Security and Privacy (SP)*, 2019.
- [30] G. I. Davida and S. M. Reddy, "Forward-Error Correction with Decision Feedback," *Information and Control*, vol. 21, no. 2, 1972.
- [31] T. J. Holman, "Error correction and detection for faults on time multiplexed data lines," Apr. 2001, U.S. Patent 6,219,817.
- [32] J. Kim, M. Sullivan, and M. Erez, "Bamboo ECC: Strong, Safe, and Flexible Codes for Reliable Computer Memory," in *Proceedings of the International Symposium on High Performance Computer Architecture (HPCA)*, 2015.

- [33] A. Singh, S. Chakravarty, G. Papadimitriou, and D. Gizopoulos, "Silent Data Errors: Sources, Detection, and Modeling," in *Proceedings of the VLSI Test Symposium (VTS)*, 2023.
- [34] H. D. Dixit, L. Boyle, G. Vunnam, S. Pendharkar, M. Beadon, and S. Sankar, "Detecting silent data corruptions in the wild," *arXiv preprint arXiv:2203.08989*, 2022.
- [35] D. Agiakatsikas, G. Papadimitriou, V. Karakostas, D. Gizopoulos, M. Psarakis, C. Belanger-Champagne, and E. Blackmore, "Impact of ´ Voltage Scaling on Soft Errors Susceptibility of Multicore Server CPUs," in *Proceedings of the 56th Annual IEEE/ACM International Symposium on Microarchitecture*, 2023.
- [36] R. W. Hamming, "Error Detecting and Error Correcting Codes," *Bell System Technical Journal*, vol. 29, no. 2, 1950.
- [37] M. Y. Hsiao, "A Class of Optimal Minimum Odd-weight-column SEC-DED Codes," *IBM Journal of Research and Development*, vol. 14, no. 4, 1970.
- [38] R. C. Bose and D. K. Ray-Chaudhuri, "On A Class of Error Correcting Binary Group Codes," *Information and Control*, vol. 3, no. 1, 1960.
- [39] I. S. Reed and G. Solomon, "Polynomial Codes over Certain Finite Fields," *Journal of the Society for Industrial and Applied Mathematics*, vol. 8, no. 2, 1960.
- [40] A. N. Udipi, N. Muralimanohar, R. Balsubramonian, A. Davis, and N. P. Jouppi, "LOT-ECC: Localized and Tiered Reliability Mechanisms for Commodity Memory Systems," in *Proceedings of the International Symposium on Computer Architecture (ISCA)*, 2012.
- [41] S. Jeong, S. Kang, and J.-S. Yang, "PAIR: Pin-aligned In-DRAM ECC architecture using expandability of Reed-Solomon code," in *Proceedings of the Design Automation Conference (DAC)*, 2020.
- [42] S. Sonawane and V. S. Baste, "Implementation of RS-CC Encoder and Decoder using MATLAB," *International Journal of Science Technology and Engineering*, vol. 5, 2019.
- [43] C. Chen, "Error-correcting codes for semiconductor memories," in *Proceedings of the 11th annual international symposium on Computer architecture*, 1984.
- [44] M. Hsiao, W. C. Carter, J. W. Thomas, and W. R. Stringfellow, "Reliability, Availability, and Serviceability of IBM Computer Systems: A Quarter Century of Progress," *IBM Journal of Research and Development*, vol. 25, no. 5, 1981.
- [45] Synopsys, "Error Correction Code (ECC) in DDR Memories," https: //www.synopsys.com/articles/ecc-memory-error-correction.html, 2020.
- [46] Advanced Micro Devices, Inc., *BIOS and Kernel Developer's Guide (BKDG) for AMD Family 15h Models 00h-0Fh Processors*, 2013.
- [47] C. Li, Y. Zhang, J. Wang, H. Chen, X. Liu, T. Huang, L. Peng, S. Zhou, L. Wang, and S. Ge, "From Correctable Memory Errors to Uncorrectable Memory Errors: What Error Bits Tell," in *Proceedings of the International Conference on High Performance Computing, Networking, Storage and Analysis (SC)*, 2022.
- [48] X. Du, C. Li, S. Zhou, X. Liu, X. Xu, T. Wang, and S. Ge, "Fault-Aware Prediction-Guided Page Offlining for Uncorrectable Memory Error Prevention," in *Proceedings of the International Conference on Computer Design (ICCD)*, 2021.
- [49] U. Kang, H.-s. Yu, C. Park, H. Zheng, J. Halbert, K. Bains, S. Jang, and J. S. Choi, "Co-Architecting Controllers and DRAM to Enhance DRAM Process Scaling," in *The Memory Forum*, vol. 14, 2014.
- [50] I. Alam and P. Gupta, "COMET: On-die and In-controller Collaborative Memory ECC Technique for Safer and Stronger Correction of DRAM Errors," in *Proceedings of the International Conference on Dependable Systems and Networks (DSN)*, 2022.
- [51] K. C. Chun, Y. K. Kim, Y. Ryu, J. Park, C. S. Oh, Y. Y. Byun, S. Y. Kim, D. H. Shin, J. G. Lee, B.-K. Ho, M.-S. Park, S.-J. Cho, S. Woo, B. M. Moon, B. Kil, S. Ahn, J. H. Lee, S. Y. Kim, S.-K. Choi, J.-S. Jeong, S.-G. Ahn, J. Kim, J. J. Kong, K. Sohn, N. S. Kim, and J.-B. Lee, "A 16-GB 640-GB/s HBM2E DRAM with a Data-Bus Window Extension Technique and a Synergetic On-Die ECC Scheme," *IEEE Journal of Solid-State Circuits*, vol. 56, no. 1, 2020.
- [52] Y. Ryu, S.-G. Ahn, J. H. Lee, J. Park, Y. K. Kim, H. Kim, Y. G. Song, H.-W. Cho, S. Cho, S. H. Song, H. Lee, U. Shin, J. Ahn, J.-M. Ryu, S. Lee, K.-H. Lim, J. Lee, J. H. Park, J.-S. Jeong, S. Joo, D. Cho, S. Y. Kim, M. Lee, H. Kim, M. Kim, J.-S. Kim, J. Kim, H. G. Kang, M.-K. Lee, S.-R. Kim, Y.-C. Kwon, Y. Y. Byun, K. Lee, S. Park, J. Youn, M.-O. Kim, K. Sohn, S.-J. Hwang, and J. Lee, "A 16 GB 1024 GB/s HBM3 DRAM With Source-Synchronized Bus Design and On-Die Error Control Scheme for Enhanced RAS Features," *IEEE Journal of Solid-State Circuits*, vol. 58, no. 4, 2023.

- [53] T.-Y. Oh, H. Chung, J.-Y. Park, K.-W. Lee, S. Oh, S.-Y. Doo, H.-J. Kim, C. Lee, H.-R. Kim, J.-H. Lee, J.-I. Lee, K.-S. Ha, Y. Choi, Y.- C. Cho, Y.-C. Bae, T. Jang, C. Park, K. Park, S. Jang, and J. S. Choi, "A 3.2 Gbps/pin 8 Gbit 1.0 V LPDDR4 SDRAM With Integrated ECC Engine for Sub-1 V DRAM Core Operation," *IEEE Journal of Solid-State Circuits*, vol. 50, no. 1, 2014.
- [54] M.-J. Park, J. Lee, K. Cho, J. Park, J. Moon, S.-H. Lee, T.-K. Kim, S. Oh, S. Choi, Y. Choi *et al.*, "A 192-Gb 12-High 896-GB/s HBM3 DRAM With a TSV Auto-Calibration Scheme and Machine-Learning-Based Layout Optimization," *IEEE Journal of Solid-State Circuits*, vol. 58, no. 1, 2022.
- [55] J. Kim, M. Sullivan, S. Lym, and M. Erez, "All-Inclusive ECC: Thorough End-to-End Protection for Reliable Computer Memory," in *Proceedings of the International Symposium on Computer Architecture (ISCA)*, 2016.
- [56] H. Wang, Y. Li, X. Zhang, X. Zhao, H. Sun, and T. Zhang, "On the Use of DRAM with Unrepaired Weak Cells in Computing Systems," in *Proceedings of the Second International Symposium on Memory Systems*, 2016.
- [57] W. W. Peterson and D. T. Brown, "Cyclic Codes for Error Detection," *Proceedings of the IRE*, vol. 49, no. 1, 1961.
- [58] R. Wu, S. Zhou, J. Lu, Z. Shen, Z. Xu, J. Shu, K. Yang, F. Lin, and Y. Zhang, "Removing Obstacles before Breaking Through the Memory Wall: A Close Look at HBM Errors in the Field," in *Proceedings of the USENIX Annual Technical Conference (USENIX)*, 2024.
- [59] X. Du and C. Li, "Predicting Uncorrectable Memory Errors from the Correctable Error History: No Free Predictors in the Field," in *Proceedings of the International Symposium on Memory Systems*, 2021.
- [60] A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Yang, A. Fan *et al.*, "The Llama 3 Herd of Models," *arXiv preprint arXiv:2407.21783*, 2024.
- [61] J. Meza, Q. Wu, S. Kumar, and O. Mutlu, "Revisiting Memory Errors in Large-Scale Production Data Centers: Analysis and Modeling of New Trends from the Field," in *Proceedings of the International Conference on Dependable Systems and Networks (DSN)*, 2015.
- [62] M. Nicolaidis, "Design for Soft Error Mitigation," *IEEE Transactions on Device and Materials Reliability*, vol. 5, no. 3, 2005.
- [63] M. V. Beigi, Y. Cao, G. Tsai, S. Gurumurthi, and V. Sridharan, "DDR5 DRAM Faults in the Field," in *Proceedings of the International Conference on Dependable Systems and Networks-Supplemental Volume (DSN-S)*, 2025.
- [64] H. Jeon, G. H. Loh, and M. Annavaram, "Efficient RAS Support for Diestacked DRAM," in *Proceedings of the International Test Conference (ITC)*, 2014.
- [65] P. J. Nair, V. Sridharan, and M. K. Qureshi, "XED: Exposing On-Die Error Detection Information for Strong Memory Reliability," in *Proceedings of the International Symposium on Computer Architecture (ISCA)*, 2016.
- [66] J. B. Halbert, K. S. Bains, and K. E. Criss, "Memory device on-die error checking and correcting code," US Patent, Nov. 2017, issued Nov. 14, 2017. [Online]. Available: https://patents.google.com/patent/ US9817714B2/en
- [67] JEDEC standard, "High Bandwidth Memory DRAM (HBM3)," in *JESD238*, 2022.
- [68] S.-L. Gong, J. Kim, S. Lym, M. Sullivan, H. David, and M. Erez, "DUO: Exposing On-Chip Redundancy to Rank-Level ECC for High Reliability," in *Proceedings of the International Symposium on High Performance Computer Architecture (HPCA)*, 2018.
- [69] G. Jung, H. J. Na, S.-H. Kim, and J. Kim, "Dual-Axis ECC: Vertical and Horizontal Error Correction for Storage and Transfer Errors," in *Proceedings of the International Conference on Computer Design (ICCD)*, 2024.
- [70] M. Patel, G. F. de Oliveira, and O. Mutlu, "HARP: Practically and Effectively Identifying Uncorrectable Errors in Memory Chips that Use On-Die Error-Correcting Codes," in *Proceedings of the International Symposium on Microarchitecture (MICRO)*, 2021.
- [71] JEDEC standard, "Low Power Double Data Rate (LPDDR) 5/5X," in *JESD209-5C*, 2023.
- [72] D. Sarwate and N. Shanbhag, "High-Speed Architectures for Reed–Solomon Decoders," *IEEE Transactions on Very Large Scale Integration (VLSI) Systems*, vol. 9, no. 5, 2001.
- [73] L. Saiz, J. Gracia, D. Gil, J.-C. Baraza-Calvo, and P. Gil-Vicente, "Reducing the Overhead of BCH Codes: New Double Error Correction Codes," *Electronics*, vol. 9, 2020.

- [74] V. Sridharan, N. DeBardeleben, S. Blanchard, K. B. Ferreira, J. Stearley, J. Shalf, and S. Gurumurthi, "Memory Errors in Modern Systems: The Good, The Bad, and The Ugly," *ACM SIGARCH Computer Architecture News*, vol. 50, no. 4, 2015.
- [75] M. Khairy, Z. Shen, T. M. Aamodt, and T. G. Rogers, "Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling," in *Proceedings of the 47th Annual International Symposium on Computer Architecture (ISCA)*, 2020.
- [76] S. Che, J. W. Sheaffer, M. Boyer, L. G. Szafaryn, L. Wang, and K. Skadron, "A Characterization of the Rodinia Benchmark Suite with Comparison to Contemporary CMP Workloads," in *Proceedings of the International Symposium on Workload Characterization (IISWC)*, 2010.
- [77] J. A. Stratton, C. I. Rodrigues, I.-J. Sung, N. Obeid, L.-W. Chang, N. Anssari, G. Liu, and W. mei W. Hwu, "Parboil: A Revised Benchmark Suite for Scientific and Commercial Throughput Computing," *Center for Reliable and High-Performance Computing*, vol. 127, no. 7.2, 2012.
- [78] L. Nai, Y. Xia, I. G. Tanase, H. Kim, and C.-Y. Lin, "GraphBIG: Understanding Graph Computing in the Context of Industrial Solutions," in *Proceedings of the International Conference on High Performance Computing, Networking, Storage and Analysis (SC)*, 2015.
- [79] M. A. Abella-Gonzalez, P. Carollo-Fern ´ andez, L.-N. Pouchet, ´ F. Rastello, and G. Rodr´ıguez, "PolyBench/Python: Benchmarking Python Environments with Polyhedral Optimizations," in *Proceedings of the 30th ACM SIGPLAN International Conference on Compiler Construction*, 2021.
- [80] S. Cha, O. Seongil, H. Shin, S. Hwang, K. Park, S. J. Jang, J. S. Choi, G. Y. Jin, Y. H. Son, H. Cho, J. H. Ahn, and N. S. Kim, "Defect Analysis and Cost-Effective Resilience Architecture for Future DRAM Devices," in *Proceedings of the International Symposium on High Performance Computer Architecture (HPCA)*, 2017.
- [81] Samsung Electronics, *16Gb HBM Flashbolt*, 2021.
- [82] Micron, "TN-40-07: Calculating memory power for DDR4 SDRAM," 2018.
- [83] N. Chatterjee, M. O'Connor, D. Lee, D. R. Johnson, S. W. Keckler, M. Rhu, and W. J. Dally, "Architecting an Energy-Efficient DRAM System for GPUs," in *Proceedings of the International Symposium on High Performance Computer Architecture (HPCA)*, 2017.
- [84] S. Yun, K. Kyung, J. Cho, J. Choi, J. Kim, B. Kim, S. Lee, K. Sohn, and J. H. Ahn, "Duplex: A Device for Large Language Models with Mixture of Experts, Grouped Query Attention, and Continuous Batching," in *Proceedings of the International Symposium on Microarchitecture (MICRO)*, 2024.
- [85] A. Jarmusch and S. Chandrasekaran, "Microbenchmarking NVIDIA's Blackwell Architecture: An in-depth Architectural Analysis," 2026. [Online]. Available: https://arxiv.org/abs/2512.02189# IV. PRIOR WORK

The inefficiency and limited reliability of multi-layer memory protection in commercial DRAMs have inspired numerous academic efforts to redesign ECC architectures. Prior work largely follows two directions: (i) schemes that consolidate protection into a strengthened S-ECC, and (ii) schemes that explicitly coordinate layers and leverage cross-layer information. This section summarizes representative research along these lines.

## *A. Stronger System ECC*

- *1) DUO:* DUO [68] bypasses on-die ECC and repurposes its internal redundancy at the system level. By forwarding redundant bits to the host through additional transfer beats, DUO extends S-ECC into a longer symbol-based codeword (e.g., RS(76,64)), providing SDDC protection. A portion of this redundancy is reserved as *on-chip redundancy parity*, which supports burst-erasure correction and verification, enabling recovery even when a complete chip failure coincides with multiple hard defects (Fig. 3a).
- *2) Unity ECC:* Unity ECC [22] presents a single-layer S-ECC framework capable of correcting both double-bit errors and single-chip failures. By allowing S-ECC to handle both frequent bit faults and rare device-level errors, Unity ECC

eliminates the need for on-die correction. This simplification reduces latency, power, and area overheads associated with O-ECC while maintaining robust end-to-end protection (Fig. 3b).

*3) Dual-Axis ECC:* Dual-Axis ECC [69] protects both storage and transfer errors via two-orientation decoding: vertical symbols correct storage faults, while horizontal overlays correct common DQS-induced transfer errors by reusing spare syndromes, eliminating L-ECC overhead and reducing retransmissions without extra redundancy.

## *B. Cross-layer Collaboration*

- *1) XED:* XED [65] proposes cooperative interaction between S-ECC and O-ECC to achieve SDDC protection. It exposes on-die ECC detection outcomes to the system through a predefined *catch-word* interface, enabling S-ECC to perform erasure-based decoding with improved correction efficiency in configurations with multiple DRAM devices per channel.
- *2) HARP:* HARP [70] improves reliability through profiling-based repair under O-ECC. It leverages O-ECC's information (e.g., via decode-bypass and/or correction-event reporting) to profile vulnerable locations and then applies repair actions to mitigate future errors at those locations. Its benefit depends on profiling coverage and available repair resources.

Table I summarizes industry practice and two representative academic designs (DUO and Unity ECC) for single-deviceper-channel organizations. Collectively, these academic approaches demonstrate the potential of single-layer ECC to improve system reliability and reduce redundancy. However, they overlook a critical industry requirement—*error concealment within DRAM devices*. To preserve product quality perception, DRAM vendors deliberately mask internal fault behavior, reporting only a limited set of vulnerable regions rather than exposing raw error counts [71]. Although academic proposals enhance transparency and end-to-end reliability, their exposure of device-level errors to the host conflicts with this industrial practice, raising concerns over warranty obligations and vendor accountability.

## V. CERBERUS

This section introduces Cerberus, a unified, *cross-layer* ECC framework that delivers high reliability at low overhead with robust end-to-end protection. Unlike conventional multilayer schemes in which each layer is designed and operated independently, Cerberus *co-designs* the generator and paritycheck matrices across layers to ensure interoperability. This interoperability allows a single encoder to produce redundancy shared across layers—each serving a distinct purpose (link protection, error concealment, or ultra-high reliability)—while preventing destructive cross-layer interference (e.g., avoiding miscorrections that amplify errors for the next layer). The following sections present the Cerberus architecture and its operational flow.

## *A. Architecture*

Cerberus targets single-device memory channels in which the channel transfer unit matches the device's internal access unit (e.g., HBM, LPDDR). In this paper, we focus on an HBM configuration with a 256-bit (32B) access unit and a total redundancy budget of 12.5% (4B per 32B data block). The 32B access unit aligns with current HBM practice, whereas the redundancy budget is lower than the HBM baseline (18.8%). Despite using less redundancy, Cerberus delivers stronger protection, making it suitable for custom HBMs and future HBM generations.

Fig. 4 illustrates the overall architecture. The framework consists of a single shared encoder ( 1 ) and three layerspecific decoders ( 1 , 2 , 3 ) that cooperate along the data path. The 32-bit redundancy generated once by the encoder is reused—wholly or partially—by the following decoding layers: (1) the *Link Layer* (LL), which provides early detection of write-path link errors; (2) the *Device Layer* (DL), which performs on-die error correction and concealment; and (3) the *System Layer* (SL), which ensures strong end-to-end, system-level reliability. Each layer interprets the shared 32 bit redundancy according to its role. The LL utilizes 16 bits exclusively for error detection; the DL reuses that same 16-bit portion for bit-level single-error correction (SEC) within the die; and the SL leverages the full 32 bits to perform symbolbased correction and detection. This *Encode-Once, Decode-Many* (EODM) organization with hierarchical redundancy reuse eliminates repeated encoding stages, reduces latency, storage overheads, and preserves seamless protection coverage across all layers without reliability gaps.

## *B. Cerberus Operations*

*1) Write Operation:* On a write, the encoder ( 1 ) takes 256-bit user data D and appends 32 bits of redundancy composed of R<sup>1</sup> and R<sup>2</sup> (16 bits each). This redundancy is produced by a generator matrix G<sup>S</sup>-ECC, which is a product of two sub-matrices, G<sup>1</sup> and G<sup>2</sup> (G<sup>S</sup>-ECC = G<sup>1</sup> · G2). G<sup>1</sup> first maps 256-bit D to a 272-bit intermediate codeword (D+R1); G<sup>2</sup> then maps this 272-bit word to a 288-bit final codeword ((D+R1)+R2). Although the description separates these as two encoding stages for conceptual clarity, practical

![](_page_6_Figure_7.jpeg)

Fig. 4: An overview of Cerberus

implementations can perform both mappings in a single step using the composite matrix GS-ECC.

The host omits conventional write-path L-ECC and transmits the 288-bit codeword ((D+R1)+R2) directly to DRAM. On the DRAM side, the first decoder ( 1 ) replaces L-ECC by verifying the link integrity of ((D+R1)+R2) using R<sup>2</sup> via the parity-check matrix H<sup>2</sup> (the dual of G2). Upon a mismatch, the decoder does not attempt correction but instead requests retransmission (e.g., via a conventional ALERT signal). The 16 bit redundancy provides early detection of transfer errors with a detection capability comparable to CRC16. If verification succeeds, the validated codeword ((D+R1)+R2) is stored in DRAM with its redundancy preserved for subsequent deviceand system-level reuse.

*2) Read Operation:* On a read, the second decoder ( 2 ) operates inside the bank group to perform *on-die* error correction and concealment. The device validates ((D+R1)+R2) using the same parity-check matrix H2, but this time, it applies the correction (SEC) when needed. This ensures that singlebit errors are corrected within the device, preventing their propagation to the memory controller.

Importantly, H<sup>2</sup> is designed to satisfy a *bounded-fault* constraint at the 16-bit symbol granularity: any miscorrection that arises from multiple flipped bits is *confined* to the originally faulty 16-bit symbol and cannot corrupt additional symbols. Consequently, device-level actions never increase symbol-level error severity. After local correction, the DRAM bypasses read-path L-ECC generation and instead forwards the corrected 288-bit codeword ((D+R1)+R2) to the controller.

At the controller, the third decoder ( 3 ) provides systemlevel protection using both redundancy fields (R1, R2). Leveraging the full 32-bit redundancy, it delivers SSC+DEC capability: it corrects either a single 16-bit symbol error (e.g., from a device-level miscorrection or critical peripheral-circuit faults) or two bit errors located in distinct symbols (e.g., overlapping storage/transfer errors). For more severe patterns, it still provides safe detection with probability 99.97%. It also extends correction beyond bank-group boundaries and replaces read-side L-ECC. When the decoder detects an uncorrectable error, it issues a single retry. If the retry is still uncorrectable, it reports a DUE. Otherwise, it treats the first event as a transient read-link or peripheral error and forwards the retry result, as in conventional L-ECC retry.

Collectively, Cerberus delivers SSC+DEC correction to the system, SEC correction to DRAM devices, and CRC16-level detection for link protection, meeting the distinct requirements of DRAM reliability and manufacturability. Compared to typical HBM designs, Cerberus offers stronger end-to-end correction at 12.5% overhead (vs. 18.8%) and is not confined to a single bank group. Compared to LPDDR-style protection, it matches the 12.5% overhead but upgrades from bit-level SEC to symbol-aware SSC+DEC at the host, while retaining on-die SEC and link detection for full end-to-end coverage.

## *C. Cross-Layer ECC Collaboration*

This architecture raises a key question: how can the system and device layers collaborate so that redundancy provisioned for the system layer is reusable in the device layer? We address this question in the following subsections by describing our device/link-layer, system-layer, and cross-layer ECC designs.

*1) Device/Link Layer:* We begin by describing the decoding mechanism at the device and link layers. These layers receive 288-bit data ((D+R1)+R2) and use R<sup>2</sup> to detect and correct errors in D and R1, resulting in a parity-check matrix H<sup>2</sup> with dimensions 16 × 288.

The device layer targets SEC-DED with a bounded-fault constraint at 16-bit granularity. We choose SEC based on observations from HBM3/4, where strong on-die correction can trigger severe miscorrections and still leave the system exposed to faults outside the bank group. Such miscorrections then force the system layer to provision additional correction strength and redundancy. For example, an O-ECC using 8-bit SSC with 2-symbol redundancy can miscorrect a two-symbol error into a three-symbol corruption, which would require 6 symbol redundancy in S-ECC for full recovery. Moreover, because error sources are not confined to a single bank group, strong intra-bank protection is helpful but offers limited reliability beyond the bank group. Therefore, we design the device layer to handle frequent small-scale bit errors, while the system layer is responsible for rarer but more severe faults.

To achieve SEC-DED with bounded-fault recovery, H<sup>2</sup> must satisfy the following conditions:

- Every column must be non-zero.
- SEC: All columns must be unique.
- DED: The sum of any two distinct columns must not be equal to any other column.
- Bounded fault: The sum of columns within any 16 column region must not match any column belonging to another region.

Additionally, since H<sup>2</sup> is also used in the link layer, it must be able to detect frequent transfer errors. To achieve this, we design H<sup>2</sup> to meet both the device-layer constraints and the CRC8 requirements. Unlike CRC16, CRC8 does not guarantee detection of all burst errors of length 9–16; however, the combined 16-bit redundancy still provides a random-error detection probability of 1 − 2 <sup>−</sup><sup>16</sup> (≈99.998%). Detecting 16 bit burst errors is especially important in DDR systems, where the burst length is typically 16, whereas in memory systems with shorter bursts (e.g., HBM4 with burst-8), this level of detection is less critical. To realize CRC8, H<sup>2</sup> must satisfy the condition that any eight consecutive columns are linearly independent.

As a result, we use H<sup>2</sup> to protect both the device and link layers, implementing SEC-DED with a bounded-fault property while also meeting CRC8 and providing CRC16-level detection capability.

*2) System Layer:* The system layer is responsible for ensuring end-to-end reliability by detecting and correcting errors that may have propagated through the device and link layers. This layer is designed to provide robust protection against severe, rare errors that escape the correction capabilities of the device and link layers. The system layer achieves this by leveraging the redundancy provided by the device layer (via R<sup>1</sup> and R2) to perform symbol-based error correction, leading to a parity-check matrix (HS-ECC) with dimensions of 32 × 288.

To achieve this, the system layer is designed to correct single 16-bit symbol errors or double-bit errors (SSC+DEC). This dual capability addresses both clustered errors caused by malfunctioning peripheral circuits (e.g., subwordline driver failures) and frequent, random bit errors that may occur simultaneously. The goal of the system layer is to maintain high levels of protection with minimal additional redundancy overhead, thereby ensuring end-to-end reliability without significant performance or storage costs.

To achieve SSC+DEC, HS-ECC must satisfy the following conditions:

- Every column must be non-zero.
- SSC: The sums of all symbol-aligned columns are unique.
- DEC: The sums of any two columns are unique.
- SSC+DEC: All sums from properties 2 and 3 should be unique (apart from double-bit errors in the same symbol, which are considered symbol errors).

For effective collaboration between the device/link layers and the system layer, HS-ECC must satisfy a single condition, which we discuss in the next section.

*3) Cross Layer:* This section describes how HS-ECC can be decomposed into H<sup>1</sup> and H<sup>2</sup> while encoding only once, and presents the single condition under which any 256-bit granularity DRAM can adopt the Cerberus framework. We begin by describing the shared-encoder generator matrix, G<sup>S</sup>-ECC.

We design G<sup>S</sup>-ECC to satisfy G<sup>S</sup>-ECC = G<sup>1</sup> · G2. For such a G<sup>1</sup> to exist, each row of G<sup>S</sup>-ECC must be expressible as a linear combination of the rows of G2. In other words, the row space of G<sup>S</sup>-ECC must be contained within the row space of G<sup>2</sup> (i.e., row(G<sup>S</sup>-ECC) ⊆ row(G2)). To ensure this condition in the parity-check matrix domain, we utilize the relationship between the generator matrix G and the parity-check matrix H (e.g., H2G<sup>⊤</sup> <sup>2</sup> = 0). In this domain, the condition is equivalent to the requirement that each row of H<sup>2</sup> be expressible as a linear combination of the rows of H<sup>S</sup>-ECC (i.e., row(H2) ⊆ row(H<sup>S</sup>-ECC)). If this simple condition holds, there are no additional constraints on adopting the Cerberus framework, which allows it to support a wide range of vendor-specific S-ECC schemes with high scalability.

![](_page_8_Figure_0.jpeg)

Fig. 5: The parity-check matrices of Cerberus for cross-layer design

## *D. Code Construction*

We derive an H-matrix that satisfies each layer's conditions through a two-step construction. First, we construct H<sup>2</sup> to provide SEC-DED with bounded-faults. We realize SEC-DED by assigning odd-weight columns. Instead of using a prefix region for bounded-faults, we enforce both the bounded fault and CRC8 properties by building the second half (8 columns) of each 16-column bounded region as XOR combinations of the columns in the first half (8 columns). This structure makes it easy to satisfy the CRC8 condition and also helps meet the SSC requirement of HS-ECC. Next, we map each bounded region of the binary H<sup>2</sup> to elements in GF(216) and place this symbolized H<sup>2</sup> (Fig. 5a) directly in the upper part of HS-ECC, thereby satisfying the cross-layer condition (row(H2) ⊆ row(HS-ECC)).

Second, we construct HS-ECC to satisfy both SSC and DEC. Since H<sup>2</sup> is already placed in the upper part of HS-ECC, we build the lower part using a greedy search. We randomly assign GF(216) elements to each symbol in the lower part, then binarize HS-ECC and check for syndrome overlaps to verify that the SSC and DEC conditions are met. If syndrome overlaps occur, we reconstruct the symbol with the largest number of overlaps and repeat this process until HS-ECC satisfies both SSC and DEC (Fig. 5b).

## *E. Hardware Implementation*

Cerberus largely reuses standard memory ECC primitives (e.g., an encoder and an SEC decoder) and updates only the G and H matrices (Fig. 6). On the write path, the controller encoder ( 1 ) computes R<sup>1</sup> and R<sup>2</sup> in a single pass using G<sup>S</sup>-ECC = G<sup>1</sup> · G2. It implements this multiplication with an XOR network (e.g., 8-level XOR trees). In DRAM, the first decoder ( 1 ) verifies writes by regenerating R<sup>2</sup> using a subset of the same XOR network.

On the read path, the second decoder in DRAM ( 2 ) generates a syndrome with an XOR-tree network and corrects single-bit errors based on the syndrome, which adds modest logic depth (≈ 4) to the combinational logic. It then forwards

![](_page_8_Figure_8.jpeg)

Fig. 6: The hardware implementation of Cerberus

the corrected 288-bit codeword (with redundancy retained) for end-to-end decoding. The third decoder in the controller ( 3 ) applies HS-ECC to the received 288-bit codeword to generate a 32-bit syndrome and runs SSC and DEC correctors in parallel [22]. The SSC corrector uses Chien search with a modified Berlekamp–Massey procedure [72], and the DEC corrector uses a block-pair solver [73].

Overall, the encoder and the first two decoders are on par with existing implementations. The main added complexity is the third decoder for SSC+DEC. However, it performs error *detection* within a single cycle to avoid latency increases on error-free accesses, and it completes correction within a single cycle by running SSC and DEC correction in parallel. Because errors are rare, we include only the detection latency in the performance evaluation (Section VI-B) and report the area overhead of advanced decoding (Section VI-C).

## VI. EVALUATION

We evaluate the reliability and performance impact of Cerberus and compare it with single-layer ECC and state-ofthe-art multi-layered DRAM ECC configurations.

TABLE II: A comparison of reliability against single-location error scenarios

|                   |                   |                    | Single-layer |             | Multi-layer        |                |                    |                    | Cross-layer    |                    |  |
|-------------------|-------------------|--------------------|--------------|-------------|--------------------|----------------|--------------------|--------------------|----------------|--------------------|--|
| Redundancy (%)    |                   |                    | 12.5% (32b)  | 18.8% (48b) | 12.5%              | (32b)          | 18.8%              | (48b)              | 12.5% (32b)    | 15.6% (40b)        |  |
| Error<br>Location | Error<br>Scenario | Decoding<br>Result | Unity ECC    | DUO         | LPDDR6/<br>SEC-DED | LPDDR6/<br>CRC | HBM4/<br>SEC-DED   | HBM4/<br>CRC       | Cerberus (32b) | Cerberus (40b)     |  |
|                   | SE (%)            | CE                 |              | 100.000     |                    |                |                    |                    |                |                    |  |
|                   |                   | CE                 | 100.000      |             | 0.048              | 0.024          | 100.000            |                    |                |                    |  |
|                   | 16E (%)           | DUE                | 0.000        |             | 99.563             | 99.976         | 0.000              |                    |                |                    |  |
|                   |                   | SDC                | 0.000        |             | 0.389              | 0.000          | 0.000              |                    |                |                    |  |
| In bank           |                   | CE                 | 0.003        | 1.558       | 0.003              | 0.000          | 0.003              | 0.003              | 0.003          | 0.003              |  |
| III bank          | 32E (%)           | DUE                | 99.972       | 98.407      | 99.585             | 99.999         | 99.997             | 99.997             | 99.971         | 99.997             |  |
|                   |                   | SDC                | 0.025        | 0.035       | 0.412              | 0.001          | $2 \times 10^{-5}$ | $1 \times 10^{-6}$ | 0.026          | $3 \times 10^{-4}$ |  |
|                   | SE+SE (%)         | CE                 | 100.000      |             | 10.515             | 0.000          | 4.988              | 4.940              | 100.000        |                    |  |
|                   |                   | DUE                | 0.0          |             | 89.485             | 100.000        | 94.980             | 95.060 0.000       |                |                    |  |
|                   |                   | SDC                | 0.0          | 000         | 0.000              | 0.000          | 0.032              | 0.000              | 0.0            | 000                |  |
|                   | SE (%)            | CE                 | 100.000      |             |                    |                |                    |                    |                |                    |  |
| Write             | DQE (%)           | CE                 |              | 100.000     |                    |                |                    |                    |                |                    |  |
| Link              | DQSE (%)          | CE                 | 0.000        | 0.000       | 99.998             | 99.998         | 49.994             | 49.991             |                | 999                |  |
| LIIIK             |                   | DUE                | 99.971       | 99.945      | 0.002              | 0.002          | 50.005             | 50.008             | 0.0            |                    |  |
|                   |                   | SDC                | 0.029        | 0.055       | 0.000              | 0.000          | 0.001              | 0.001              | 0.0            | 000                |  |
|                   | SE (%) CE         |                    | 100          |             | 100.000            | 0.000          | 100.000            | 0.000              |                | .000               |  |
|                   | 3E (70)           | DUE                | 0.0          |             | 0.000              | 100.000        | 0.000              | 100.000            |                | 000                |  |
|                   | DE (%)            | DE (%) CE          |              | 100.000     |                    | 0.000          |                    |                    |                | 100.000            |  |
| Out bank          |                   | DUE                | 0.000        |             |                    |                |                    | 0.0                | 0.000          |                    |  |
| (Read peri/link)  | DQE (%)           | CE                 | 100.000      |             |                    |                |                    |                    |                |                    |  |
| (redu periorine)  | DQL (70)          | DUE                |              |             |                    |                | 0.000              |                    |                |                    |  |
|                   |                   | CE                 | 0.000        | 0.000       | 99.998             | 99.998         | 49.994             | 49.991             | 99.972         | 100.000*           |  |
|                   | DQSE (%)          | DUE                | 99.971       | 99.945      | 0.002              | 0.002          | 50.005             | 50.008             | 0.000          | 0.000              |  |
|                   |                   | SDC                | 0.029        | 0.055       | 0.000              | 0.000          | 0.001              | 0.001              | 0.028          | $1 \times 10^{-4}$ |  |

<sup>\*</sup> Rounded to 100.000% for display; actual value is slightly lower (e.g., 99.9998%).

#### A. Error Coverage

We evaluate the reliability of Cerberus using Monte Carlo error-injection experiments. We inject random errors under various error scenarios, and quantify correction and detection through ECC decoding. Based on system-level analyses of DRAM faults [15]–[17], [58], [74], we derive location-specific error patterns, detailed below.

We consider three error locations: (i) *In bank*, (ii) *Write link*, and (iii) *Out bank*. First, *In bank* covers faults internal to a DRAM bank. We consider the following error scenarios: Single Error (SE; caused by cell or BLSA), 16-bit Error (16E; CSL or SWL), 32-bit Error (32E; SWD) and a combination of two Single Errors (SE+SE). Second, *Write link* denotes the transmit path during writes, including transmission-induced faults. We select error scenarios: SE, Data Pin Error (DQE), and Data Strobe Error (DQSE). Third, *Out bank* spans the read path beyond the bank—device periphery and I/O—and we select error scenarios: SE, Double Error (DE; peripheral errors (e.g., TSV)) and DQE, DQSE (read-link errors).

For each error scenario, we inject errors at the specified DRAM location and flip the designated number of bits, each independently with a 50% probability. We evaluate both single-location and multi-location cases. Each ECC scheme classifies outcomes as correctable (CE), detectable but uncorrectable (DUE), or undetectable (SDC), and we aggregate results over 10 million iterations. For link errors, we count DUEs as CEs due to retransmission and retry.

We compare Cerberus against a range of layered ECC configurations. For single-layer schemes, we use Unity ECC [22] and DUO [68], both deployed on single-device memory. For multi-layer schemes, we evaluate state-of-the-art DRAM configurations, LPDDR6 [25] and HBM4 [26], which employ S-ECC as either SEC-DED or CRC [37], [52]. Finally, we

evaluate the cross-layer framework Cerberus with both 12.5% (32b) and 15.6% (40b) redundancy.

1) Single-Location: Table II summarizes error coverage and redundancy for each ECC configuration under single-location scenarios. For the *In bank*, all configurations correct SE and 16E at 100% except for 16E in LPDDR6. Because LPDDR6 partitions the 12.5% redundancy per layer, it lacks sufficient budget to correct 16E. Moreover, LPDDR6/SEC-DED shows a high SDC rate for 16E and 32E due to O-ECC miscorrections. By contrast, Cerberus guarantees 100% correction of 16E while using the same 12.5% total redundancy budget across layers. For other scenarios, Cerberus provides strong detection capability for 32E and, with increased redundancy, can further enhance robustness.

For SE+SE, none of the multi-layer configurations provides guaranteed correction. In LPDDR6, the SEC-DED O-ECC detects the event but forwards the uncorrected data to the controller without a hint, while the downstream S-ECC (SEC-DED or CRC) lacks sufficient correction capability (except in cases where errors occur in on-die parity bits that are not transferred). Similarly, in HBM4, the SSC O-ECC corrects only when both errors fall within the same symbol; otherwise the residual pattern exceeds the S-ECC capability. This behavior aligns with prior observations on multi-layer protection [50], although the SDC rates differ because [50] considers SEC O-ECC (without double-error detection). Unity ECC and DUO correct SE+SE by concentrating redundancy in a strengthened single layer that supports double-bit correction. Cerberus also corrects SE+SE via its SSC+DEC while preserving on-die error concealment for the common case of single-bit errors.

For the *Write link*, Unity ECC and DUO can correct SE and DQE with strong S-ECC, but wider transfer errors (DQSE) are only *detected*, not corrected. Because they are single-layer

TABLE III: A comparison of reliability against multi-location error scenarios

|                   |                   |                    | Single-layer |             | Multi-layer        |                |                     |                    | Cross-layer    |                    |  |
|-------------------|-------------------|--------------------|--------------|-------------|--------------------|----------------|---------------------|--------------------|----------------|--------------------|--|
| Redundancy (%)    |                   |                    | 32b (12.5%)  | 48b (18.8%) | 32b (12.5%)        |                | 48b (18.8%)         |                    | 32b (12.5%)    | 40b (15.6%)        |  |
| Error<br>Location | Error<br>Scenario | Decoding<br>Result | Unity ECC    | DUO         | LPDDR6/<br>SEC-DED | LPDDR6/<br>CRC | HBM4/<br>SEC-DED    | HBM4/<br>CRC       | Cerberus (32b) | Cerberus (40b)     |  |
|                   | SE+               | CE                 | 100.000      |             | 100.000            | 0.000          | 100.000             | 0.000              | 100.000        |                    |  |
| In bank+          | SE (%)            | DUE                | 0.000        |             | 0.000              | 100.000        | 0.000               | 100.000            | 0.000          |                    |  |
| Out bank          | SE+               | CE                 | 0.947        | 100.000     |                    | 0.0            | 0.000 100.000       |                    |                |                    |  |
| Out bank          | DE (%)            | DUE                | 99.019       | 0.000       |                    | 100            | 0.000               |                    |                | 000                |  |
|                   | DE (%)            | SDC                | 0.034        | 0.000       |                    | 0.0            | 0.000               |                    |                | 000                |  |
|                   | 16E+<br>DQE (%)   | CE                 | 5.545        | 100.000     | 0.047              | 0.025          |                     |                    | 100.000        |                    |  |
|                   |                   | DUE                | 94.431       | 0.000       | 99.565             | 99.975         | 0.000               |                    |                |                    |  |
| In bank+          |                   | SDC                | 0.024        | 0.000       | 0.388              | 0.000          | 0.000               |                    |                |                    |  |
| Write Link        | 32E+<br>DQSE (%)  | CE                 | 0.000        | 0.000       | 0.002              | 0.000          | 0.001               | 0.001              | 0.003          | 0.003              |  |
|                   |                   | DUE                | 99.972       | 99.950      | 99.586             | 99.999         | 99.999              | 99.999             | 99.972         | 99.997             |  |
|                   |                   | SDC                | 0.028        | 0.050       | 0.412              | 0.001          | $2 \times 10^{-5}$  | $8 \times 10^{-7}$ | 0.025          | $3 \times 10^{-4}$ |  |
|                   | SE+               | CE                 | 1.293        | 100.000     | 100.000            | 0.000          | 100.000             | 0.000              | 100.000        |                    |  |
| In bank+          | SE+               | DUE                | 98.672       | 0.000       | 0.000              | 100.000        | 0.000 100.000 0.000 |                    | 000            |                    |  |
| Out bank+         | SE (%)            | SDC                | 0.035        | 0.000       | 0.000              | 0.000          | 0.000               | 0.000 0.000        |                | 000                |  |
| Write Link        | SE+               | CE                 | 0.000        | 0.000       | 99.998             | 99.998         | 50.010              | 49.982             | 99.998         | 99.999             |  |
| WITH LINK         | DQE+              | DUE                | 97.971       | 99.950      | 0.002              | 0.002          | 49.988              | 50.017             | 0.002          | 0.001              |  |
|                   | DQSE (%)          | SDC                | 0.028        | 0.050       | $3 \times 10^{-5}$ | 0.000          | 0.001               | 0.001              | 0.000          | 0.000              |  |

schemes, they cannot perform early detection, so corrupted data may be written back uncorrected, allowing subsequent faults to accumulate and increase the risk of severe reliability issues. In contrast, the multi-layer baselines employ L-ECC and thus offer high detection for SE/DQE/DQSE, enabling correction via retransmission. However, when the L-ECC is provisioned with limited redundancy, as in HBM4, DQSE detection drops to roughly half of cases. Cerberus, on the other hand, guarantees 100% detection (and thus correction) for SE and DQE, and for wider DQSE, the first decoder already provides high detection; any remaining cases are caught by the stronger third decoder, yielding a robust end-to-end design.

For the *Out bank*, Unity ECC and DUO can correct SE, DE, and DQE using a strong S-ECC. However, they only *detect* DQSE because they do not guarantee a retry after detection. In contrast, the multi-layer configurations do not guarantee correction for peripheral DE. Moreover, when they apply S-ECC with CRC (LPDDR6-CRC and HBM4-CRC), they fail to correct even SE and instead provide detection only. This limitation stems from redundancy partitioning across layers, which prevents redundancy reuse. For example, HBM4 can allocate 32b to a strong O-ECC (SSC) and leave only 16b for S-ECC. This allocation forces weaker codes (SEC-DED or CRC) and leaves the system vulnerable to out-of-bank errors. In contrast, Cerberus guarantees 100% correction for SE, DE, and DQE, and achieves near-complete correction for DQSE with retries.

2) Multi-Location: Table III summarizes the error coverage and redundancy of each ECC configuration under multi-location scenarios. For combined *In bank* and *Out bank* errors, Unity ECC and DUO guarantee 100% correction for SE+SE. However, for SE+DE, DUO still corrects while Unity ECC fails. Although Unity ECC corrects SE and DE individually in the single-location case, the overlap across two locations exposes the limitation of relying solely on S-ECC. The multi-layer approaches show the same behavior as in the single-location *Out bank* case (e.g., failing to correct peripheral DE), because although O-ECC corrects *In bank* SE, *Out bank* errors still remain due to the weaker code used for S-ECC.

In contrast, Cerberus guarantees correction for both SE+SE and SE+DE across locations even with a small redundancy budget, as O-ECC corrects the In bank SE and S-ECC covers all Out bank errors. For combined In bank and Write link, configurations with 12.5% redundancy (Unity ECC/LPDDR6) do not guarantee correction for 16E+DQE. However, Cerberus can correct it even with the same redundancy. For 32E+DQSE, Cerberus likewise provides significantly higher detection than other schemes with the same redundancy. Finally, in the highrisk scenario where errors occur simultaneously at all locations (In bank, Out bank, and Write link), single-layer configurations reveal the limitation of relying only on S-ECC, since all overlapping errors are exposed to the system layer. DUO can correct SE+SE+SE with its stronger scheme, but it fails once larger errors are involved, and Unity ECC also cannot guarantee correction across these cases. Multi-layer configurations improve locality by letting each layer handle its corresponding errors (e.g., O-ECC handles In bank errors and L-ECC handles link errors), but because the limited redundancy budget is divided across layers, they still fail to provide correction in all cases. In contrast, Cerberus retains the role of each layer while reusing redundancy across layers, enabling nearly 100% correction across all cases with a small redundancy budget and thus offering robust end-to-end reliability.

Rather than limiting our evaluation to in-DRAM faults, we assess reliability using error scenarios that span the entire memory system, and show that Cerberus maintains strong reliability under both single- and multi-location errors. In addition, Cerberus is a scalable framework that can accommodate higher redundancy. Although this increases overhead, it improves detection capability and yields a more robust system.

#### B. Performance & Energy Consumption

1) GPU Performance: We evaluate the performance impact of Cerberus (32b) and Cerberus (40b) on GPUs using the cycle-level simulator Accel-Sim [75], and compare it against HBM4 [26], Unity ECC [22], and DUO [68]. Our system model is based on an NVIDIA V100 GPU configured with 32 HBM channels, with detailed parameters listed in Table IV.

![](_page_11_Figure_0.jpeg)

Fig. 7: Comparison of GPU performance and DRAM energy for Cerberus across the evaluated benchmarks

TABLE IV: The simulation configuration

| Components | Configuration                                               |  |  |  |  |
|------------|-------------------------------------------------------------|--|--|--|--|
| # of SMs   | 80                                                          |  |  |  |  |
|            | 1132 MHz, 4 warp schedulers/SM,                             |  |  |  |  |
| SM         | up to 32 blocks/SM, up to 48 warps/SM                       |  |  |  |  |
| L1 cache   | Up to 128KiB, 4 banks, 128B line, 256-way, 384 MSHR entries |  |  |  |  |
| L2 cache   | 4MiB, 128B lines, 16-way, 192 MSHR entries                  |  |  |  |  |
| Memory     | 256B channel interleaving, 64-entry scheduling queue,       |  |  |  |  |
| controller | FR-FCFS scheduling                                          |  |  |  |  |
|            | HBM4, 32 channels, 6.4Gbps                                  |  |  |  |  |
| Memory     | tRCD = 30-cycle, tRRD = 4-cycle, CL = 24-cycle,             |  |  |  |  |
|            | tCCDS = 2-cycle, tCCDL = 4-cycle, WL = 14-cycle             |  |  |  |  |

To capture a wide range of application behaviors, we use 16 workloads drawn from four benchmark suites: Rodinia [76], Parboil [77], GraphBIG [78], and PolyBench [79].

The ECC decoder affects the time from a read command to the output of the first corresponding data beat (tCL), while the encoder impacts the write latency (tWL) [80]. We evaluate the performance by adjusting these two timing parameters for each ECC configuration. For HBM4, we estimate the encoder and decoder latency overheads of 16-bit SSC-based O-ECC as 2ns and 5ns, respectively [80]. In contrast, the latency overheads of both S-ECC (CRC) and L-ECC (parity) are less than 1ns for both encoding and decoding. For Unity ECC and DUO, we derive their timing parameters by first removing the O-ECC latency overhead from the HBM4 baseline and then adding each scheme's synthesized latency. Logic synthesis in a UMC 28nm process yields latency overheads of 1.46ns for Unity ECC and 1.92ns for DUO. For both Cerberus (32b) and Cerberus (40b), we remove the O-ECC encoder latency and reduce the decoder latency from 5ns to 2ns under the modified O-ECC scheme [80], since the two configurations have nearly identical O-ECC latency. We also model S-ECC latency from logic synthesis, adding 0.85ns for Cerberus (32b) and 0.89ns for Cerberus (40b). Finally, we convert all latency values into clock cycles at 1.6 GHz (6.4 Gbps after QDR in HBM4).

Fig. 7a reports the instructions per cycle (IPC) for HBM4, Unity ECC, DUO, Cerberus (32b) and Cerberus (40b), normalized to HBM4. Across benchmarks, Cerberus (32b) improves IPC by 0.2%, 0.2%, 1.1%, and 1.4% (0.7% geomean), while Cerberus (40b) improves IPC by 0.3%, 0.1%, 1.2%, and 0.4% (0.5% geomean). Unity ECC and DUO achieve similar IPC gains, but they provide lower reliability than Cerberus. In contrast, Cerberus delivers higher reliability than HBM4 while using less redundancy, yet still improves IPC. It also attains performance comparable to Unity ECC (0.9% geomean). These benefits stem from the EODM organization of Cerberus, which eliminates repeated encoding stages through efficient reuse of redundancy across layers.

*2) DRAM Energy Consumption:* Cerberus uses different storage and transfer bit widths than HBM4. To estimate the resulting DRAM power/energy, we use HBM2E operating currents from a datasheet [81].

We assume the precharge standby current (IDD2N) is independent of bit width, while the incremental activation current (IDD0−IDD2N) and active-standby current (IDD3N−IDD2N) scale with the stored bit count (e.g., (256+32+16) bits in HBM4 vs. (256+32) bits in Cerberus (32b)). For read/write activity, we partition the incremen-

TABLE V: Estimated DRAM operating currents (per pseudochannel)

| DRAM current        | Unity<br>ECC | DUO      | HBM4   | Cerberus<br>(32b) | Cerberus<br>(40b) |  |  |  |  |
|---------------------|--------------|----------|--------|-------------------|-------------------|--|--|--|--|
| IDD0 (mA)           | 47.42        | 47.56    | 47.56  | 47.42             | 47.49             |  |  |  |  |
| (ACT-PRE)           | (99.7%)      | (100%)   | (100%) | (99.7%)           | (99.9%)           |  |  |  |  |
| IDD2N (mA)          |              | 44.88    |        |                   |                   |  |  |  |  |
| (Precharge standby) | (100%)       |          |        |                   |                   |  |  |  |  |
| IDD3N (mA)          | 47.69        | 47.84    | 47.84  | 47.69             | 47.77             |  |  |  |  |
| (Active standby)    | (99.7%)      | (100%)   | (100%) | (99.7%)           | (99.8%)           |  |  |  |  |
| IDD4R (mA)          | 521.28       | 547.75   | 525.72 | 521.28            | 534.52            |  |  |  |  |
| (Read)              | (99.2%)      | (104.2%) | (100%) | (99.2%)           | (101.7%)          |  |  |  |  |
| IDD4W (mA)          | 365.27       | 383.07   | 368.25 | 365.29            | 374.17            |  |  |  |  |
| (Write)             | (99.2%)      | (104%)   | (100%) | (99.2%)           | (101.6%)          |  |  |  |  |

tal currents (IDD4R-IDD2N and IDD4W-IDD2N) between bank-group-internal transfer (cells $\rightarrow$ O-ECC) and bank-group-external transfer (O-ECC $\rightarrow$ processor) and apply a 61%:39% split from prior HBM2 analysis [8]. We then scale each component by the corresponding transfer width: HBM4 transfers (256+32+16) bits within a bank group and (256+16) bits outside, whereas Cerberus (32b) transfers (256+32) bits within a bank group and (256+32) bits within a bank group and (256+32) bits outside. Finally, we compute overall DRAM energy consumption using the Micron DDR4 power calculator [82] with these current values. Table V summarizes the DRAM operating currents used in our evaluation.

Fig. 7b shows the results. Cerberus (32b) reduces energy by 1.84% on average compared to HBM4. This is primarily because intra-die transfers from cells to bank peripherals often consume more energy than off-chip transfer [8], [83], and Cerberus reduces these bank-group-internal transfers. With higher redundancy, Cerberus (40b) consumes 0.86% more energy than HBM4 on average to provide stronger protection.

## C. Hardware Overheads

To estimate the hardware costs, we implement SystemVerilog models for the encoder and decoders. We synthesize these models with Synopsys Design Compiler using a UMC 28nm standard library. We then normalize the resulting area to NAND2 equivalents (the number of NAND2 gates that occupy the same area) to present process-independent results.

Table VI summarizes the area cost of Cerberus. Decoder 1 (L-ECC) and Decoder 2 (O-ECC) reside inside the DRAM device. Together, they require only 14,693 NAND2 equivalents in Cerberus (32b) and 15,662 NAND2 equivalents in Cerberus (40b), which correspond to 0.0074 mm² and 0.0079 mm², respectively. This overhead is negligible compared to an HBM stack footprint (e.g., 121 mm² for HBM3 [84]). Decoder 3 (S-ECC) dominates the overall overhead, mainly due to the DEC corrector. Even so, the processor-side overheads (encoder+S-ECC) total 127,583 NAND2 equivalents for Cerberus (32b) and 167,954 NAND2 equivalents for Cerberus (40b). Relative to modern GPUs with billions of transistors, this is a tiny fraction of the transistor count (e.g.,  $2.5 \times 10^{-6}$  to  $3.2 \times 10^{-6}$  of a 208B-transistor Blackwell [85]).

TABLE VI: Area overheads (in NAND2 equivalents)

|                      | Cerberus (32b)                      | Cerberus (40b)                      |
|----------------------|-------------------------------------|-------------------------------------|
| Encoder              | $1632.79 \ \mu \text{m}^2 \ (3240)$ | $2015.49 \ \mu \text{m}^2 \ (3999)$ |
| Decoder 1<br>(L-ECC) | 1205.40 $\mu \text{m}^2$ (2392)     | 1398.43 $\mu \text{m}^2$ (2775)     |
| Decoder 2<br>(O-ECC) | 6199.87 $\mu \text{m}^2$ (12301)    | 6495.15 $\mu \text{m}^2$ (12887)    |
| Decoder 3<br>(S-ECC) | 62669.04 $\mu \text{m}^2$ (124343)  | 82633.15 $\mu \text{m}^2$ (163955)  |

#### VII. CONCLUSION

This paper presents Cerberus, a cross-layer ECC co-design that addresses key challenges of multi-layer ECC: inefficient use of redundancy, overlapping protection coverage, and destructive cross-layer interference due to miscorrections. Implemented on HBM4 with a cross-layer ECC design, Cerberus reduces redundancy by 33.3% while still providing higher reliability through efficient redundancy reuse. Moreover, its Encode-Once, Decode-Many (EODM) architecture eliminates unnecessary encoding stages, improving performance and delivering seamless coverage without protection gaps. Overall, Cerberus provides a promising framework for achieving high reliability in future HBM- and LPDDR-based systems.

#### REFERENCES

- A. Spessot and H. Oh, "1T-1C Dynamic Random Access Memory Status, Challenges, and Prospects," *IEEE Transactions on Electron Devices*, vol. 67, no. 4, 2020.
- [2] S.-L. Gong, J. Kim, and M. Erez, "DRAM Scaling Error Evaluation Model Using Various Retention Time," in *Proceedings of the Annual IEEE/IFIP International Conference on Dependable Systems and Networks Workshops (DSN-W)*, 2017.
- [3] O. Mutlu, Main Memory Scaling: Challenges and Solution Directions. Springer New York, 2015.
- [4] H. Hassan, M. Patel, J. S. Kim, A. G. Yaglikci, N. Vijaykumar, N. M. Ghiasi, S. Ghose, and O. Mutlu, "CROW: A Low-Cost Substrate for Improving DRAM Performance, Energy Efficiency, and Reliability," in Proceedings of the 46th international symposium on computer architecture. 2019.
- [5] H. Ha, Understanding and Improving the Energy Efficiency of DRAM. Stanford University, 2018.
- [6] H. Park, S.-M. Yu, and J. Song, "An 11 Gb/s 0.376 pJ/Bit Capacitor-Less Dicode Transceiver With Pattern-Dependent Equalizations TIA Termination for Parallel DRAM Interfaces," *IEEE Access*, 2024.
- [7] Y. Jung, S. Lee, H. Kim, and S. Cho, "A Supply-Noise-Induced Jitter-Cancelling Clock Distribution Network for LPDDR5 Mobile DRAM featuring a 2nd-order Adaptive Filter," in *Proceedings of the International Solid State Circuits Conference (ISSCC)*, vol. 65, 2022.
- [8] M. O'Connor, N. Chatterjee, D. Lee, J. Wilson, A. Agrawal, S. W. Keckler, and W. J. Dally, "Fine-Grained DRAM: Energy-Efficient DRAM for Extreme Bandwidth Systems," in *Proceedings of the 50th Annual IEEE/ACM International Symposium on Microarchitecture*, 2017.
- [9] K. S. Yim, C. Pham, M. Saleheen, Z. Kalbarczyk, and R. Iyer, "Hauberk: Lightweight Silent Data Corruption Error Detector for GPGPU," in Proceedings of the International Symposium on Parallel and Distributed Processing (IPDPS), 2011.
- [10] D. Fiala, F. Mueller, C. Engelmann, R. Riesen, K. Ferreira, and R. Brightwell, "Detection and Correction of Silent Data Corruption for Large-Scale High-Performance Computing," in *Proceedings of the International Conference on High Performance Computing, Networking, Storage and Analysis (SC)*, 2012.
- [11] R. Yeleswarapu and A. K. Somani, "Addressing multiple bit/symbol errors in DRAM subsystem," *PeerJ Computer Science*, vol. 7, 2021.

- [12] M. B. Sullivan, M. T. I. Ziad, A. Jaleel, and S. W. Keckler, "Implicit Memory Tagging: No-Overhead Memory Safety Using Alias-Free Tagged ECC," in *Proceedings of the International Symposium on Computer Architecture (ISCA)*, 2023.
- [13] S. Li, D. H. Yoon, K. Chen, J. Zhao, J. H. Ahn, J. B. Brockman, Y. Xie, and N. P. Jouppi, "MAGE: Adaptive Granularity and ECC for Resilient and Power Efficient Memory Systems," in *Proceedings of the International Conference on High Performance Computing, Networking, Storage and Analysis*, 2012.
- [14] B. Salami, O. S. Unsal, and A. C. Kestelman, "Evaluating Built-In ECC of FPGA On-Chip Memories for the Mitigation of Undervolting Faults," in *Proceedings of the 27th Euromicro International Conference on Parallel, Distributed and Network-Based Processing (PDP)*, 2019.
- [15] M. V. Beigi, Y. Cao, S. Gurumurthi, C. Recchia, A. Walton, and V. Sridharan, "A Systematic Study of DDR4 DRAM Faults in the Field," in *Proceedings of the International Symposium on High Performance Computer Architecture (HPCA)*, 2023.
- [16] H. Chung, E. Oh, S. Baek, H. Yoon, J. Yoo, S. Lee, Y. Lee, A. Bramhanand, B. Dodds, Y. Zhou, and N. S. Kim, "DRAM Fault Classification through Large-Scale Field Monitoring for Robust Memory RAS Management," in *Proceedings of the International Symposium on Microarchitecture (MICRO)*, 2025.
- [17] J. Jung and M. Erez, "Predicting Future-System Reliability with a Component-Level DRAM Fault Model," in *Proceedings of the 56th Annual IEEE/ACM International Symposium on Microarchitecture*, 2023.
- [18] A. Bacchini, M. Rovatti, G. Furano, and M. Ottavi, "Characterization of Data Retention Faults in DRAM Devices," in *Proceedings of the International Symposium on Defect and Fault Tolerance in VLSI Systems (DFT)*, 2014.
- [19] Y. Moon, S. Shin, S. Jang, D. Won, and S. Kang, "A Novel Prediction-Based Two-Tiered ECC for Mitigating SWD Errors in HBM," *IEEE Transactions on Very Large Scale Integration (VLSI) Systems*, 2024.
- [20] M. Patel, J. S. Kim, H. Hassan, and O. Mutlu, "Understanding and Modeling On-Die Error Correction in Modern DRAM: An Experimental Study using Real Devices," in *Proceedings of the International Conference on Dependable Systems and Networks (DSN)*, 2019.
- [21] M. Patel, J. S. Kim, T. Shahroodi, H. Hassan, and O. Mutlu, "Bit-Exact ECC Recovery (BEER): Determining DRAM On-Die ECC Functions by Exploiting DRAM Data Retention Characteristics," in *Proceedings of the International Symposium on Microarchitecture (MICRO)*, 2020.
- [22] D. Kim, J. Lee, W. Jung, M. B. Sullivan, and J. Kim, "Unity ECC: Unified Memory Protection Against Bit and Chip Errors," in *Proceedings of the International Conference on High Performance Computing, Networking, Storage and Analysis (SC)*, 2023.
- [23] R. Xie, A. U. Haq, Y. Fang, L. Ma, S. Sen, S. Venkataramani, L. Liu, and T. Zhang, "Breaking the HBM Bit Cost Barrier: Domain-Specific ECC for AI Inference Infrastructure," *IEEE Computer Architecture Letters*, 2025.
- [24] JEDEC standard, "Double Data Rate (DDR) 5," in *JESD79-5C.01*, 2024.
- [25] ——, "LPDDR6 standard," in *JESD209-6*, 2024.
- [26] ——, "High Bandwidth Memory (HBM4) DRAM," in *JESD270-4*, 2025.
- [27] D. H. Yoon and M. Erez, "Virtualized and Flexible ECC for Main Memory," in *Proceedings of the fifteenth International Conference on Architectural support for programming languages and operating systems*, 2010.
- [28] K. Criss, K. Bains, R. Agarwal, T. Bennett, T. Grunzke, J. K. Kim, H. Chung, and M. Jung, "Improving Memory Reliability by Bounding DRAM Faults: DDR5 improved reliability features," in *Proceedings of the International Symposium on Memory Systems (MEMSYS)*, 2020.
- [29] L. Cojocar, K. Razavi, C. Giuffrida, and H. Bos, "Exploiting Correcting Codes: On the Effectiveness of ECC Memory Against Rowhammer Attacks," in *Proceedings of the IEEE Symposium on Security and Privacy (SP)*, 2019.
- [30] G. I. Davida and S. M. Reddy, "Forward-Error Correction with Decision Feedback," *Information and Control*, vol. 21, no. 2, 1972.
- [31] T. J. Holman, "Error correction and detection for faults on time multiplexed data lines," Apr. 2001, U.S. Patent 6,219,817.
- [32] J. Kim, M. Sullivan, and M. Erez, "Bamboo ECC: Strong, Safe, and Flexible Codes for Reliable Computer Memory," in *Proceedings of the International Symposium on High Performance Computer Architecture (HPCA)*, 2015.

- [33] A. Singh, S. Chakravarty, G. Papadimitriou, and D. Gizopoulos, "Silent Data Errors: Sources, Detection, and Modeling," in *Proceedings of the VLSI Test Symposium (VTS)*, 2023.
- [34] H. D. Dixit, L. Boyle, G. Vunnam, S. Pendharkar, M. Beadon, and S. Sankar, "Detecting silent data corruptions in the wild," *arXiv preprint arXiv:2203.08989*, 2022.
- [35] D. Agiakatsikas, G. Papadimitriou, V. Karakostas, D. Gizopoulos, M. Psarakis, C. Belanger-Champagne, and E. Blackmore, "Impact of ´ Voltage Scaling on Soft Errors Susceptibility of Multicore Server CPUs," in *Proceedings of the 56th Annual IEEE/ACM International Symposium on Microarchitecture*, 2023.
- [36] R. W. Hamming, "Error Detecting and Error Correcting Codes," *Bell System Technical Journal*, vol. 29, no. 2, 1950.
- [37] M. Y. Hsiao, "A Class of Optimal Minimum Odd-weight-column SEC-DED Codes," *IBM Journal of Research and Development*, vol. 14, no. 4, 1970.
- [38] R. C. Bose and D. K. Ray-Chaudhuri, "On A Class of Error Correcting Binary Group Codes," *Information and Control*, vol. 3, no. 1, 1960.
- [39] I. S. Reed and G. Solomon, "Polynomial Codes over Certain Finite Fields," *Journal of the Society for Industrial and Applied Mathematics*, vol. 8, no. 2, 1960.
- [40] A. N. Udipi, N. Muralimanohar, R. Balsubramonian, A. Davis, and N. P. Jouppi, "LOT-ECC: Localized and Tiered Reliability Mechanisms for Commodity Memory Systems," in *Proceedings of the International Symposium on Computer Architecture (ISCA)*, 2012.
- [41] S. Jeong, S. Kang, and J.-S. Yang, "PAIR: Pin-aligned In-DRAM ECC architecture using expandability of Reed-Solomon code," in *Proceedings of the Design Automation Conference (DAC)*, 2020.
- [42] S. Sonawane and V. S. Baste, "Implementation of RS-CC Encoder and Decoder using MATLAB," *International Journal of Science Technology and Engineering*, vol. 5, 2019.
- [43] C. Chen, "Error-correcting codes for semiconductor memories," in *Proceedings of the 11th annual international symposium on Computer architecture*, 1984.
- [44] M. Hsiao, W. C. Carter, J. W. Thomas, and W. R. Stringfellow, "Reliability, Availability, and Serviceability of IBM Computer Systems: A Quarter Century of Progress," *IBM Journal of Research and Development*, vol. 25, no. 5, 1981.
- [45] Synopsys, "Error Correction Code (ECC) in DDR Memories," https: //www.synopsys.com/articles/ecc-memory-error-correction.html, 2020.
- [46] Advanced Micro Devices, Inc., *BIOS and Kernel Developer's Guide (BKDG) for AMD Family 15h Models 00h-0Fh Processors*, 2013.
- [47] C. Li, Y. Zhang, J. Wang, H. Chen, X. Liu, T. Huang, L. Peng, S. Zhou, L. Wang, and S. Ge, "From Correctable Memory Errors to Uncorrectable Memory Errors: What Error Bits Tell," in *Proceedings of the International Conference on High Performance Computing, Networking, Storage and Analysis (SC)*, 2022.
- [48] X. Du, C. Li, S. Zhou, X. Liu, X. Xu, T. Wang, and S. Ge, "Fault-Aware Prediction-Guided Page Offlining for Uncorrectable Memory Error Prevention," in *Proceedings of the International Conference on Computer Design (ICCD)*, 2021.
- [49] U. Kang, H.-s. Yu, C. Park, H. Zheng, J. Halbert, K. Bains, S. Jang, and J. S. Choi, "Co-Architecting Controllers and DRAM to Enhance DRAM Process Scaling," in *The Memory Forum*, vol. 14, 2014.
- [50] I. Alam and P. Gupta, "COMET: On-die and In-controller Collaborative Memory ECC Technique for Safer and Stronger Correction of DRAM Errors," in *Proceedings of the International Conference on Dependable Systems and Networks (DSN)*, 2022.
- [51] K. C. Chun, Y. K. Kim, Y. Ryu, J. Park, C. S. Oh, Y. Y. Byun, S. Y. Kim, D. H. Shin, J. G. Lee, B.-K. Ho, M.-S. Park, S.-J. Cho, S. Woo, B. M. Moon, B. Kil, S. Ahn, J. H. Lee, S. Y. Kim, S.-K. Choi, J.-S. Jeong, S.-G. Ahn, J. Kim, J. J. Kong, K. Sohn, N. S. Kim, and J.-B. Lee, "A 16-GB 640-GB/s HBM2E DRAM with a Data-Bus Window Extension Technique and a Synergetic On-Die ECC Scheme," *IEEE Journal of Solid-State Circuits*, vol. 56, no. 1, 2020.
- [52] Y. Ryu, S.-G. Ahn, J. H. Lee, J. Park, Y. K. Kim, H. Kim, Y. G. Song, H.-W. Cho, S. Cho, S. H. Song, H. Lee, U. Shin, J. Ahn, J.-M. Ryu, S. Lee, K.-H. Lim, J. Lee, J. H. Park, J.-S. Jeong, S. Joo, D. Cho, S. Y. Kim, M. Lee, H. Kim, M. Kim, J.-S. Kim, J. Kim, H. G. Kang, M.-K. Lee, S.-R. Kim, Y.-C. Kwon, Y. Y. Byun, K. Lee, S. Park, J. Youn, M.-O. Kim, K. Sohn, S.-J. Hwang, and J. Lee, "A 16 GB 1024 GB/s HBM3 DRAM With Source-Synchronized Bus Design and On-Die Error Control Scheme for Enhanced RAS Features," *IEEE Journal of Solid-State Circuits*, vol. 58, no. 4, 2023.

- [53] T.-Y. Oh, H. Chung, J.-Y. Park, K.-W. Lee, S. Oh, S.-Y. Doo, H.-J. Kim, C. Lee, H.-R. Kim, J.-H. Lee, J.-I. Lee, K.-S. Ha, Y. Choi, Y.- C. Cho, Y.-C. Bae, T. Jang, C. Park, K. Park, S. Jang, and J. S. Choi, "A 3.2 Gbps/pin 8 Gbit 1.0 V LPDDR4 SDRAM With Integrated ECC Engine for Sub-1 V DRAM Core Operation," *IEEE Journal of Solid-State Circuits*, vol. 50, no. 1, 2014.
- [54] M.-J. Park, J. Lee, K. Cho, J. Park, J. Moon, S.-H. Lee, T.-K. Kim, S. Oh, S. Choi, Y. Choi *et al.*, "A 192-Gb 12-High 896-GB/s HBM3 DRAM With a TSV Auto-Calibration Scheme and Machine-Learning-Based Layout Optimization," *IEEE Journal of Solid-State Circuits*, vol. 58, no. 1, 2022.
- [55] J. Kim, M. Sullivan, S. Lym, and M. Erez, "All-Inclusive ECC: Thorough End-to-End Protection for Reliable Computer Memory," in *Proceedings of the International Symposium on Computer Architecture (ISCA)*, 2016.
- [56] H. Wang, Y. Li, X. Zhang, X. Zhao, H. Sun, and T. Zhang, "On the Use of DRAM with Unrepaired Weak Cells in Computing Systems," in *Proceedings of the Second International Symposium on Memory Systems*, 2016.
- [57] W. W. Peterson and D. T. Brown, "Cyclic Codes for Error Detection," *Proceedings of the IRE*, vol. 49, no. 1, 1961.
- [58] R. Wu, S. Zhou, J. Lu, Z. Shen, Z. Xu, J. Shu, K. Yang, F. Lin, and Y. Zhang, "Removing Obstacles before Breaking Through the Memory Wall: A Close Look at HBM Errors in the Field," in *Proceedings of the USENIX Annual Technical Conference (USENIX)*, 2024.
- [59] X. Du and C. Li, "Predicting Uncorrectable Memory Errors from the Correctable Error History: No Free Predictors in the Field," in *Proceedings of the International Symposium on Memory Systems*, 2021.
- [60] A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Yang, A. Fan *et al.*, "The Llama 3 Herd of Models," *arXiv preprint arXiv:2407.21783*, 2024.
- [61] J. Meza, Q. Wu, S. Kumar, and O. Mutlu, "Revisiting Memory Errors in Large-Scale Production Data Centers: Analysis and Modeling of New Trends from the Field," in *Proceedings of the International Conference on Dependable Systems and Networks (DSN)*, 2015.
- [62] M. Nicolaidis, "Design for Soft Error Mitigation," *IEEE Transactions on Device and Materials Reliability*, vol. 5, no. 3, 2005.
- [63] M. V. Beigi, Y. Cao, G. Tsai, S. Gurumurthi, and V. Sridharan, "DDR5 DRAM Faults in the Field," in *Proceedings of the International Conference on Dependable Systems and Networks-Supplemental Volume (DSN-S)*, 2025.
- [64] H. Jeon, G. H. Loh, and M. Annavaram, "Efficient RAS Support for Diestacked DRAM," in *Proceedings of the International Test Conference (ITC)*, 2014.
- [65] P. J. Nair, V. Sridharan, and M. K. Qureshi, "XED: Exposing On-Die Error Detection Information for Strong Memory Reliability," in *Proceedings of the International Symposium on Computer Architecture (ISCA)*, 2016.
- [66] J. B. Halbert, K. S. Bains, and K. E. Criss, "Memory device on-die error checking and correcting code," US Patent, Nov. 2017, issued Nov. 14, 2017. [Online]. Available: https://patents.google.com/patent/ US9817714B2/en
- [67] JEDEC standard, "High Bandwidth Memory DRAM (HBM3)," in *JESD238*, 2022.
- [68] S.-L. Gong, J. Kim, S. Lym, M. Sullivan, H. David, and M. Erez, "DUO: Exposing On-Chip Redundancy to Rank-Level ECC for High Reliability," in *Proceedings of the International Symposium on High Performance Computer Architecture (HPCA)*, 2018.
- [69] G. Jung, H. J. Na, S.-H. Kim, and J. Kim, "Dual-Axis ECC: Vertical and Horizontal Error Correction for Storage and Transfer Errors," in *Proceedings of the International Conference on Computer Design (ICCD)*, 2024.
- [70] M. Patel, G. F. de Oliveira, and O. Mutlu, "HARP: Practically and Effectively Identifying Uncorrectable Errors in Memory Chips that Use On-Die Error-Correcting Codes," in *Proceedings of the International Symposium on Microarchitecture (MICRO)*, 2021.
- [71] JEDEC standard, "Low Power Double Data Rate (LPDDR) 5/5X," in *JESD209-5C*, 2023.
- [72] D. Sarwate and N. Shanbhag, "High-Speed Architectures for Reed–Solomon Decoders," *IEEE Transactions on Very Large Scale Integration (VLSI) Systems*, vol. 9, no. 5, 2001.
- [73] L. Saiz, J. Gracia, D. Gil, J.-C. Baraza-Calvo, and P. Gil-Vicente, "Reducing the Overhead of BCH Codes: New Double Error Correction Codes," *Electronics*, vol. 9, 2020.

- [74] V. Sridharan, N. DeBardeleben, S. Blanchard, K. B. Ferreira, J. Stearley, J. Shalf, and S. Gurumurthi, "Memory Errors in Modern Systems: The Good, The Bad, and The Ugly," *ACM SIGARCH Computer Architecture News*, vol. 50, no. 4, 2015.
- [75] M. Khairy, Z. Shen, T. M. Aamodt, and T. G. Rogers, "Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling," in *Proceedings of the 47th Annual International Symposium on Computer Architecture (ISCA)*, 2020.
- [76] S. Che, J. W. Sheaffer, M. Boyer, L. G. Szafaryn, L. Wang, and K. Skadron, "A Characterization of the Rodinia Benchmark Suite with Comparison to Contemporary CMP Workloads," in *Proceedings of the International Symposium on Workload Characterization (IISWC)*, 2010.
- [77] J. A. Stratton, C. I. Rodrigues, I.-J. Sung, N. Obeid, L.-W. Chang, N. Anssari, G. Liu, and W. mei W. Hwu, "Parboil: A Revised Benchmark Suite for Scientific and Commercial Throughput Computing," *Center for Reliable and High-Performance Computing*, vol. 127, no. 7.2, 2012.
- [78] L. Nai, Y. Xia, I. G. Tanase, H. Kim, and C.-Y. Lin, "GraphBIG: Understanding Graph Computing in the Context of Industrial Solutions," in *Proceedings of the International Conference on High Performance Computing, Networking, Storage and Analysis (SC)*, 2015.
- [79] M. A. Abella-Gonzalez, P. Carollo-Fern ´ andez, L.-N. Pouchet, ´ F. Rastello, and G. Rodr´ıguez, "PolyBench/Python: Benchmarking Python Environments with Polyhedral Optimizations," in *Proceedings of the 30th ACM SIGPLAN International Conference on Compiler Construction*, 2021.
- [80] S. Cha, O. Seongil, H. Shin, S. Hwang, K. Park, S. J. Jang, J. S. Choi, G. Y. Jin, Y. H. Son, H. Cho, J. H. Ahn, and N. S. Kim, "Defect Analysis and Cost-Effective Resilience Architecture for Future DRAM Devices," in *Proceedings of the International Symposium on High Performance Computer Architecture (HPCA)*, 2017.
- [81] Samsung Electronics, *16Gb HBM Flashbolt*, 2021.
- [82] Micron, "TN-40-07: Calculating memory power for DDR4 SDRAM," 2018.
- [83] N. Chatterjee, M. O'Connor, D. Lee, D. R. Johnson, S. W. Keckler, M. Rhu, and W. J. Dally, "Architecting an Energy-Efficient DRAM System for GPUs," in *Proceedings of the International Symposium on High Performance Computer Architecture (HPCA)*, 2017.
- [84] S. Yun, K. Kyung, J. Cho, J. Choi, J. Kim, B. Kim, S. Lee, K. Sohn, and J. H. Ahn, "Duplex: A Device for Large Language Models with Mixture of Experts, Grouped Query Attention, and Continuous Batching," in *Proceedings of the International Symposium on Microarchitecture (MICRO)*, 2024.
- [85] A. Jarmusch and S. Chandrasekaran, "Microbenchmarking NVIDIA's Blackwell Architecture: An in-depth Architectural Analysis," 2026. [Online]. Available: https://arxiv.org/abs/2512.02189