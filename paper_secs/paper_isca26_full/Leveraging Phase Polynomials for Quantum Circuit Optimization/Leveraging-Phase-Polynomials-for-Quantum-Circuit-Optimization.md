# Leveraging Phase Polynomials for Quantum Circuit Optimization

Zihan Chen Rutgers University Piscataway, NJ, USA zihan.chen.cs@rutgers.edu

Mingkuan Xu Carnegie Mellon University Pittsburgh, PA, USA mingkuan@cmu.edu

Henry Chen Rutgers University Piscataway, NJ, USA hc867@rutgers.edu

Vannessa Chan Rutgers University Piscataway, NJ, USA vlc74@rutgers.edu

Yuwei Jin Rutgers University Piscataway, NJ, USA jyw413482880@gmail.com

> Won Woo Ro Yonsei University Seoul, Korea wro@yonsei.ac.kr

Enhyeok Jang Yonsei University Seoul, Korea enhyeok.jang@yonsei.ac.kr

Eddy Z. Zhang Rutgers University Piscataway, NJ, USA eddy.zhengzhang@gmail.com

*Abstract*—Quantum circuits on resource-limited hardware require optimizing regions dominated by {CNOT, Rz}, which account for a large fraction of operations and often dominate execution cost. This optimization can be challenging because phasepolynomial blocks are fragmented by basis-changing gates such as H, and optimizing phase parities alone may increase the cost of downstream basis transformations. Existing phase-polynomial approaches are limited to single-block or phase-only optimization, while subcircuit rewriting approaches are local and scale poorly beyond small rewrite windows. We introduce *PhasePoly*, a compiler optimization pass that jointly optimizes phase-parity and output-parity networks and employs a cross-block intermediate representation to reuse parities across phase-polynomial block barriers. This approach is effective because its unified paritymatrix representation exposes long-range {CNOT, Rz} structure that local rewriting and single-block methods cannot capture. *PhasePoly* reduces total gate count by up to 50.00% (34.70% on average) and CNOT count by up to 48.57% (26.83% on average), while scaling to large circuits and improving both fault-tolerant compilation and near-term hardware execution. *PhasePoly* is available at [https://github.com/ruadapt/PhasePoly.](https://github.com/ruadapt/PhasePoly)

*Index Terms*—Phase Polynomial Optimization, Quantum Circuit Optimization, Cross-block Intermediate Representation.

