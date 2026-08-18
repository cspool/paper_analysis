# <span id="page-1-1"></span><span id="page-1-0"></span>*A. Zero-Knowledge Proofs*

A zero-knowledge proof (ZKP) is a cryptographic protocol in which a *prover* can convince a *verifier* about the truth of a computational *statement* on some private *witness* data, without revealing the witness (i.e., *zero-knowledge*). Modern applications rely on *succinct* ZKPs, known as zk-SNARKs (Succinct Non-Interactive Arguments of Knowledge) [\[9\]](#page-13-14), [\[28\]](#page-13-15), [\[56\]](#page-14-10) or zk-STARKs (Scalable Transparent Arguments of Knowledge) [\[7\]](#page-13-16). In these protocols, the proof size is typically polylogarithmic in the problem size or even constant, and the verification is exceptionally fast (e.g., a few milliseconds). This succinctness, however, comes at the cost of high computational overheads for the prover. Therefore, we focus on *proof generation* in this work, and adopt the standard three-layer view of modern ZKPs [\[10\]](#page-13-17), [\[15\]](#page-13-18): *arithmetization*, *polynomial interactive oracle proof (PIOP)*, and *polynomial commitment scheme (PCS)*.

Firstly, arithmetization reduces the statement's computation to a set of algebraic constraints over a finite field. There are two common forms. (1) The *Rank-1 Constraint System (R1CS)* [\[8\]](#page-13-19) flattens the entire inputs, outputs, and intermediate values of the computation (represented as a "circuit") into a single witness vector w. Each constraint (i.e., a circuit "gate") enforces a quadratic relationship ⟨a,w⟩· ⟨b,w⟩ = ⟨c,w⟩, where ⟨·,·⟩ denotes an inner product, and a, b, c are constant coefficient vectors. (2) The *Plonkish* arithmetization [\[25\]](#page-13-20) arranges the variables from w into a table (or "trace"), where each table row *i* corresponds to a gate and has several "wire" variables representing the gate's inputs (*wa*,*<sup>i</sup>* ,*wb*,*i*) and output (*wc*,*i*). The gate enforces the constraint *qL*,*iwa*,*i*+*qR*,*iwb*,*i*+*qO*,*iwc*,*i*+*qM*,*i*(*wa*,*iwb*,*i*)+*qC*,*<sup>i</sup>* = 0 where the constant coefficients are called "selectors". The consistency between wires in different rows (e.g., the output *wc*,*<sup>i</sup>* of a gate being the input *wa*, *<sup>j</sup>* of another) requires another set of *permutation (wiring) constraints*.

Then, a polynomial interactive oracle proof (PIOP) phase transforms the constraints into an interactive verification protocol. The prover commits the witness to a set of polynomials, and the verifier queries a small number of random evaluations to check the low-degree relations, permutation constraints, and product/zero properties, with high soundness [\[76\]](#page-14-11). The prover-side polynomial computations are often performed in two ways: (1) using the *number theoretic transform (NTT)* to switch between the coefficient and evaluation domains, turning polynomial multiplications into element-wise operations [\[28\]](#page-13-15), [\[69\]](#page-14-12); or (2) using the multilinear extension (MLE) over the Boolean hypercube, combined with the *sumcheck* protocol [\[47\]](#page-13-13), to avoid costly large NTTs.

Lastly, a polynomial commitment scheme (PCS) binds the prover to the polynomials and later opens selected evaluations with succinct proofs. This step, combined with the Fiat-Shamir transform [\[23\]](#page-13-21), allows the above PIOP to become noninteractive. PCS has two popular instantiations. (1) *Pairingbased* schemes (e.g., KZG [\[38\]](#page-13-22)) require a trusted setup but provide constant-size proofs and extremely fast verification. A typical construction uses elliptic curves (ECs) with a special multi-scalar multiplication (MSM) kernel [\[53\]](#page-14-13). (2) *Hashbased* schemes (e.g., FRI [\[6\]](#page-13-23)) realize commitments via Merkle trees [\[52\]](#page-14-14), yielding transparent (i.e., no trusted setup) systems at the cost of larger, logarithmically-sized proofs and hashheavy verification [\[7\]](#page-13-16), [\[58\]](#page-14-3).

Sparsity. ZKP circuits commonly exhibit structural *sparsity* that hardware acceleration can exploit. It arises from two primary sources: (1) control selectors (*qL*,*qR*,*qM*,*qO*) are

TABLE I REPRESENTATIVE ZKP PROTOCOLS AND THEIR KEY CHARACTERISTICS AND IMPLEMENTATIONS.

<span id="page-2-0"></span>

| Protocol   | Trusted<br>Setup | Proof<br>Size | Prover<br>Time | Verifier<br>Time | Arithmetization | PIOP              | PCS | Field      | Bitwidth      |
|------------|------------------|---------------|----------------|------------------|-----------------|-------------------|-----|------------|---------------|
| Groth16    | Per-circuit      | Small const.  | Long           | Short            | R1CS            | Linear PCP (NTT)  | KZG | EC         | 256, 384, 768 |
| HyperPlonk | Universal        | Medium        | Medium         | Medium           | Plonkish        | HyperPlonk (MLE)  | KZG | EC         | 256, 384      |
| Plonky2    | None             | Large         | Short          | Long             | Plonkish        | PLONK-style (NTT) | FRI | Goldilocks | 64            |

<span id="page-2-1"></span>TABLE II EXECUTION TIME PERCENTAGE AND ARITHMETIC INTENSITY (MODMULS/ELEMENT) OF MAJOR KERNELS IN ZKP PROTOCOLS.

| Kernel      | Groth16 | HyperPlonk | Plonky2 | Arith. Intensity |
|-------------|---------|------------|---------|------------------|
| NTT         | 29.32%  | -          | 0.15%   | 10               |
| MSM         | 69.73%  | 59.34%     | -       | 170              |
| Sumcheck    | -       | 33.49%     | -       | 12               |
| Merkle tree | -       | -          | 68.84%  | 191              |
| Polynomial  | 0.96%   | 7.14%      | 14.17%  | 0.5              |
| Other Hash  | -       | 0.03%      | 0.02%   | 191              |

inherently binary in Plonkish arithmetization, and (2) nonarithmetic operations such as comparison and bitwise logic (XOR, AND, OR) decompose field elements into individual bits, each occupying a full witness cell. These properties persist across circuits regardless of specific applications.

## <span id="page-2-2"></span>*B. Representative Protocols and Characteristics*

We select three representative protocols summarized in [Table I,](#page-2-0) capturing the diversity of modern ZKPs. They exhibit tradeoffs in trusted setup, proof size, prover time, verifier time, etc., and thus are used in different applications.

Groth16 [\[30\]](#page-13-5) is a highly optimized zk-SNARK for R1CS. We follow its popular instantiation, libsnark [\[69\]](#page-14-12). It uses R1CS and pairing-based KZG commitments with ECs. It requires a trusted setup *for each circuit*, which is inconvenient. The proof consists of three group elements and can be verified via three pairings, offering very fast verification. The proof size is typically a few hundred bytes. Groth16 has been widely adopted in early ZKP applications, such as anonymous voting [\[32\]](#page-13-2), decentralized file storage [\[24\]](#page-13-1), and privacy-preserving blockchains like Zcash [\[68\]](#page-14-15).

HyperPlonk [\[13\]](#page-13-6) uses Plonkish arithmetization, and MLEbased PIOP with sumchecks. This avoids large NTTs and enables fast, near-linear-time provers. It uses a *universal* (circuit-agnostic) trusted setup. In our baseline implementation [\[21\]](#page-13-24), the PCS is instantiated with KZG. The typical reported proof sizes are on the order of a few kilobytes. Liu et al. [\[44\]](#page-13-25) proposed a scalable collaborative zk-SNARK running on multiple servers, based on the HyperPlonk protocol.

Plonky2 [\[58\]](#page-14-3) also adopts PLONK-style arithmetization and PIOP, but uses an *FRI-based* PCS, yielding a *transparent setup*. It targets fast recursion over the 64-bit Goldilocks field. Plonky2 exposes several protocol-level parameters to allow explicit tradeoffs between the prover time, proof size, and security. It also supports recursive composition to further shrink the proof size and verifier time. Plonky2 has been adopted in modern ZKP systems, such as Ethereum-compatible zkEVM block proving [\[75\]](#page-14-16), [\[86\]](#page-14-17) and verifiable data processing via Lagrange's zk-SQL coprocessors [\[41\]](#page-13-3).

[Table I](#page-2-0) also highlights *the diverse field choices and bitwidth requirements* across protocols. Groth16 and HyperPlonk operate over large prime fields used by ECs (e.g., BN128, BLS12- 381 [\[91\]](#page-14-18), MNT4753 [\[69\]](#page-14-12)), so their bitwidths typically range from 256 bits up to 768 bits. In contrast, Plonky2 uses the 64 bit Goldilocks field with *p* = 2 <sup>64</sup> −2 <sup>32</sup> +1 [\[58\]](#page-14-3), which greatly simplifies modular reduction on standard 64-bit hardware.

[Table II](#page-2-1) further illustrates the percentages of prover execution time spent on major computational kernels across the three protocols, including NTT, MSM, sumcheck, Merkle tree, polynomial computations, and other hash functions for Fiat-Shamir transform [\[23\]](#page-13-21). We defer the kernels' mathematical details to [Section VI.](#page-6-0) These experiments are performed on an 80-thread CPU server, using a mock circuit of 2<sup>20</sup> gates for each protocol. Further details of the baseline configurations and workloads are in [Section VII.](#page-9-0) Generally, the kernels in the PCS phase, e.g., MSM or Merkle tree, often account for the largest shares of prover time. But other kernels like NTT, sumcheck, and polynomial computations still consume notable portions. These results demonstrate *the diversity in computational kernels* of ZKP. [Table II](#page-2-1) also reports the arithmetic intensity of each kernel. The MSM and Merkle tree kernels are compute-bound, while the NTT, sumcheck, and polynomial operations are memory-bound.

Generality. Beyond the above protocols, recent hash-based schemes such as Orion [\[84\]](#page-14-19) and Spartan [\[70\]](#page-14-20) prioritize high prover throughput and transparency, but often at the expense of megabyte-scale proof sizes. While this work does not target these specific protocols directly, their underlying arithmetic primitives remain fundamentally similar. In fact, virtually all modern ZKPs follow the common three-layer construction in [Section II-A.](#page-1-0) Consequently, we believe the above polynomial and commitment primitives for PIOP and PCS, including NTT, MSM, sumcheck, and hash, constitute the computational core of both current and foreseeable future protocols.

## III. MOTIVATION

The high computational cost of proof generation has recently motivated many hardware accelerator designs for ZKPs. In this section, we briefly describe prior efforts and point out their limitations to motivate our work.

