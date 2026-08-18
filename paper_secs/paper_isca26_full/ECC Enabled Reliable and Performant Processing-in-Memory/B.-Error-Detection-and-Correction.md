# *B. Error Detection and Correction*

DRAM systems tolerate faults using error detection and correction mechanisms by adding redundancy to stored data. Data and redundancy bits are grouped into a *codeword*. Common error correcting codes (ECC) include single-error correction (SEC) codes, which correct single-bit errors, and Reed–Solomon (RS) codes, which detect and correct symbol errors. In contrast, cyclic redundancy checks (CRC) provide high-coverage error detection.

Although CRC is primarily designed for detection, an appropriately chosen CRC polynomial can also enable singlebit error correction [40]. For example, when the data length is fixed (e.g., 128-bit data with an 8-bit CRC), the decoder can map each syndrome to a unique bit position and correct a single flipped bit.

<sup>1</sup>Based on the analysis of Lee et al. [46] (subsection V-B).

![](_page_2_Figure_0.jpeg)

Fig. 2: Rank-PIM and bank-PIM with ECC engines.

## C. Processing-in-Memory (PIM) Architectures

PIMs integrate compute units within or near memory chips [4]–[6], [30], [33], [57]. <sup>2</sup> PIM designs differ primarily in the placement of compute units within the memory hierarchy. This placement directly determines internal bandwidth, peak performance, and applicable reliability mechanisms. As illustrated in Figure 2, rank-PIMs place compute units at the rank level, whereas bank-PIMs position them near individual banks on each chip.

Recent prototypes demonstrate both approaches. For rank-PIM, Samsung introduced rank-level and CXL-based PIM designs that exploit multiple ranks for bandwidth amplification [33], [72]. For bank-PIM, designs such as UPMEM [3] integrate independent data processing units near each bank. SK Hynix [43] and Samsung [48] further proposed "all-bank" PIM architectures, in which the host issues compact all-bank commands that trigger SIMD-style execution across near-bank compute units, alleviating command bandwidth constraints. These architectures support GEMV and other vector kernels using a simple command set [17], [18].

Within each bank, the PIM unit incorporates a small SRAM buffer to store operands and intermediate results, enabling arithmetic operations at the bank access bandwidth. This buffering reduces row activations and improves row buffer locality.

