# Distilling Magic States in the Bicycle Architecture

Shifan Xu\*, Kun Liu<sup>†</sup>, Patrick Rall<sup>‡</sup>, Zhiyang He<sup>§</sup>, Yongshan Ding\*<sup>†</sup>

\*Department of Applied Physics, Yale University, New Haven, CT 06511, USA

<sup>†</sup>Department of Computer Science, Yale University, New Haven, CT 06511, USA

<sup>‡</sup>IBM Quantum, IBM Research, Cambridge, MA 02142, USA

§Department of Mathematics, Massachusetts Institute of Technology, Cambridge, MA 02139, USA

Abstract—Magic State Distillation is considered to be one of the promising methods for supplying the non-Clifford resources required to achieve universal fault tolerance. Conventional MSD protocols implemented in surface codes often require multiple code blocks and lattice surgery rounds, resulting in substantial qubit overhead, especially at low target error rates.

In this work, we present practical magic state distillation factories on Bivariate Bicycle (BB) codes that execute Pauli-measurement-based Clifford circuits inside a single BB code block. We formulate distillation circuit design as a joint optimization of logical qubit mapping, gate scheduling, measurement nativization, and protocol compression via qubit recycling. Based on detailed resource analysis and simulations, our BB factories have space-time volume comparable to that of leading distillation factories while delivering lower target error at a smaller qubit footprint, and are particularly compelling as second-round distillers following magic state cultivations.

Index Terms—Magic state distillation, quantum LDPC codes, fault-tolerant quantum computing

#### I. Introduction

Building large-scale quantum computers, which can realize transformative applications such as factoring [1], requires the use of quantum error correction to suppress physical noise and enable universal, fault-tolerant quantum computation (FTQC) [2], [3]. To protect quantum information from decoherence, foundational works [4]-[7] showed that we can encode information in stabilizer codes, and perform repeated quantum measurements and classical decoding to correct from arbitrary physical errors. These schemes enabled the development of threshold theorems for FTQC [8], which states that arbitrarily large quantum computation can be realized through QEC, assuming that physical error rates are below a constant threshold. Among the many codes developed, the surface code [9]–[11] has been at the center of QEC research for the past two decades due to its promising practical performance, including low connectivity requirement, high threshold and fast decoding algorithms. Despite these advantages, surface code incurs a significant space overhead in realizing FTOC: factoring 2048-bit integers in surface code architectures uses physical qubits on the scale of millions [12], [13]. More recently, significant research has studied quantum low-density parity-check (LDPC) codes, which can realize FTQC with low space overhead [14]. This asymptotic promise led to the development of many families of OLDPC codes with practical parameters [15], notably the Bivariate Bicycle (BB) codes introduced by IBM [16].

To perform computation on encoded quantum information, many schemes and architectures have been proposed for surface codes [10], [17]-[20] and QLDPC codes [14], [21]-[27]. An essential component common to almost all existing architectures is the production of high-fidelity non-Clifford resource states known as magic states. These magic states can be consumed through gate teleportation [28] to faulttolerantly implement non-Clifford logical gates on encoded information. Conceptually, gate teleporting magic states is used in most architectures because non-Clifford gates, without which we cannot perform universal quantum computation, are at this time more costly to implement using other common mechanisms such as transversal gates. Gate teleportation serves as an approach where most of the cost of non-Clifford gates is offloaded to magic state preparation, while the active cost at runtime (namely performing the teleportation) is relatively lower. In most architectures, the total overhead and logical gate speed are often bottlenecked by the costs of the magic state factories.

The most widely applied method of magic state production is called magic state distillation (MSD) [29], which consumes many copies of noisy, lower-fidelity magic states to produce fewer copies of higher-fidelity magic states. This procedure is applied to logical qubits, and typically iterated to suppress logical error rate to the scale needed for large scale FTQC  $(\leq 10^{-12})$ . Conventional MSD factories are therefore very resource-intensive, often becoming the architectural bottleneck. An alternative approach called magic state cultivation (MSC) [30] gained recent attention due to its impressive practical performance, achieving a logical error rate of  $2 \times 10^{-9}$ at  $10^{-3}$  physical error rate. MSC injects a physical magic state into a surface code logical qubit, and suppresses infidelity by applying multiple rounds of post-selection and physical non-Clifford gates. Due to its use of exponentially scaling post-selection, MSC does not scale asymptotically and may not suffice for large scale FTQC.

In this work, we introduce new designs of magic state factories that integrate MSC on surface code and MSD in BB codes, achieving low logical error rates while using significantly fewer resources compared to conventional distillation factories. In more detail:

• We present a collection of MSD protocols that run

![](_page_1_Figure_0.jpeg)

Fig. 1: Magic-state factory design space. Target logical error rate as a function of available physical qubits for surface-code cultivation, BB-code distillation on gross and two-gross codes, and two-level protocols combining cultivation with two-gross or surface-code-only distillation at physical error rate  $p_{\rm phys}=10^{-3}$ . Our BB-based factories achieve lower output error at similar qubit budgets, and the cultivation + two-gross pipeline extends to lower error regimes than cultivation can reach.

entirely within a single BB code block<sup>1</sup>, in contrast to conventional factories which run MSD on multiple blocks of QLDPC codes or surface codes. This one block design confers significant savings in the space overhead of magic state factories, reducing the physical qubit count from thousands to hundreds. Additionally, it simplifies the architecture, decreases the number of long range interblock connections, and reduces decoding complexity.

• We present comprehensive, end-to-end optimizations for existing MSD circuits, including (i) logical qubit mapping to maximize the use of native, low cost logical gates<sup>2</sup> on the BB code, (ii) a masking technique that augments rotations with Z on inactive qubits to turn a set of sparse native logical operation on all logical qubits into a denser set over the active logical qubits, (iii) gate scheduling cast as a Traveling Salesman Problem to minimize compiled circuit depth, and (iv) parallel execution of two MSD protocols on the same BB code block.

Together, these methods significantly reduce the circuit depth of MSD and improve the factory throughput.

• We introduce a general method to compress MSD circuits to be supported on fewer logical qubits while maintaining the same level of logical error suppression. As examples, we compress the 49-to-1 protocol from 13 to 7 qubits, the 51-to-3CS protocol from 18 to 9 qubits, the 64-to-2CCZ protocol from 17 to 10 qubits, making these circuits implementable within a single BB code block. Moreover, this method is generic to any magic state distillation protocol generated by tri-orthogonal matrices. As near-term quantum computers are space-limited, our method

brings large MSD protocols significantly closer to practice.

We benchmark a collection of magic state factories

 We benchmark a collection of magic state factories with varying protocols, codes, and noise levels. Our results quantify the substantial overhead reduction enabled by the proposed optimizations, and, through sensitivity analysis, provide practical guidance for protocol design under different operating regimes, while highlighting the key bottlenecks and improvement opportunities for each design.

As shown in Figure 1, our leading proposal is to perform 15-to-1 MSD on a block of two-gross code (an instance of BB code), using logical magic states supplied by MSC on a surface code. Under  $10^{-3}$  physical error rate, this protocol is estimated to reach logical error rates lower than what is currently achievable through MSC, while using just one block of surface code and one block of two-gross code. Our factories naturally fit into the bicycle architecture [27], a promising near-term FTQC architecture based on the BB codes [16]. Conceptually, the bicycle architecture encodes information in BB codes and performs logical operations using recentlydeveloped code surgery techniques [31]–[34]. These operations consume magic states, which makes magic generation the central component that determines the overall efficiency. Our designs thereby constitute an essential improvement to the bicycle architecture. For general extractor architectures [26], our techniques can also be applied to design other efficient magic state factories.

The rest of this paper is organized as follows. Sec. II reviews the fundamentals of quantum error correction, the BB code family, and magic state distillation. In Sec. III, we present a full stack BB code based magic state distillation factory design. Sec. IV introduces multiple optimization methods to reduce both the depth and the size of the MSD circuit. Sec. V introduces a protocol-level compression method to reduce the logical-qubit footprint of the MSD circuit by recycling inactive logical qubits. In Sec. VI and Sec. VII, we evaluate the space time cost and output logical error rates of our approach and compare them with surface code based magic state distillation and cultivation.

#### II. BACKGROUND

# Distilling Magic States in the Bicycle Architecture

Shifan Xu\*, Kun Liu<sup>†</sup>, Patrick Rall<sup>‡</sup>, Zhiyang He<sup>§</sup>, Yongshan Ding\*<sup>†</sup>

\*Department of Applied Physics, Yale University, New Haven, CT 06511, USA

<sup>†</sup>Department of Computer Science, Yale University, New Haven, CT 06511, USA

<sup>‡</sup>IBM Quantum, IBM Research, Cambridge, MA 02142, USA

§Department of Mathematics, Massachusetts Institute of Technology, Cambridge, MA 02139, USA

Abstract—Magic State Distillation is considered to be one of the promising methods for supplying the non-Clifford resources required to achieve universal fault tolerance. Conventional MSD protocols implemented in surface codes often require multiple code blocks and lattice surgery rounds, resulting in substantial qubit overhead, especially at low target error rates.

In this work, we present practical magic state distillation factories on Bivariate Bicycle (BB) codes that execute Pauli-measurement-based Clifford circuits inside a single BB code block. We formulate distillation circuit design as a joint optimization of logical qubit mapping, gate scheduling, measurement nativization, and protocol compression via qubit recycling. Based on detailed resource analysis and simulations, our BB factories have space-time volume comparable to that of leading distillation factories while delivering lower target error at a smaller qubit footprint, and are particularly compelling as second-round distillers following magic state cultivations.

Index Terms—Magic state distillation, quantum LDPC codes, fault-tolerant quantum computing

#### I. Introduction

Building large-scale quantum computers, which can realize transformative applications such as factoring [1], requires the use of quantum error correction to suppress physical noise and enable universal, fault-tolerant quantum computation (FTQC) [2], [3]. To protect quantum information from decoherence, foundational works [4]-[7] showed that we can encode information in stabilizer codes, and perform repeated quantum measurements and classical decoding to correct from arbitrary physical errors. These schemes enabled the development of threshold theorems for FTQC [8], which states that arbitrarily large quantum computation can be realized through QEC, assuming that physical error rates are below a constant threshold. Among the many codes developed, the surface code [9]–[11] has been at the center of QEC research for the past two decades due to its promising practical performance, including low connectivity requirement, high threshold and fast decoding algorithms. Despite these advantages, surface code incurs a significant space overhead in realizing FTOC: factoring 2048-bit integers in surface code architectures uses physical qubits on the scale of millions [12], [13]. More recently, significant research has studied quantum low-density parity-check (LDPC) codes, which can realize FTQC with low space overhead [14]. This asymptotic promise led to the development of many families of OLDPC codes with practical parameters [15], notably the Bivariate Bicycle (BB) codes introduced by IBM [16].

To perform computation on encoded quantum information, many schemes and architectures have been proposed for surface codes [10], [17]-[20] and QLDPC codes [14], [21]-[27]. An essential component common to almost all existing architectures is the production of high-fidelity non-Clifford resource states known as magic states. These magic states can be consumed through gate teleportation [28] to faulttolerantly implement non-Clifford logical gates on encoded information. Conceptually, gate teleporting magic states is used in most architectures because non-Clifford gates, without which we cannot perform universal quantum computation, are at this time more costly to implement using other common mechanisms such as transversal gates. Gate teleportation serves as an approach where most of the cost of non-Clifford gates is offloaded to magic state preparation, while the active cost at runtime (namely performing the teleportation) is relatively lower. In most architectures, the total overhead and logical gate speed are often bottlenecked by the costs of the magic state factories.

The most widely applied method of magic state production is called magic state distillation (MSD) [29], which consumes many copies of noisy, lower-fidelity magic states to produce fewer copies of higher-fidelity magic states. This procedure is applied to logical qubits, and typically iterated to suppress logical error rate to the scale needed for large scale FTQC  $(\leq 10^{-12})$ . Conventional MSD factories are therefore very resource-intensive, often becoming the architectural bottleneck. An alternative approach called magic state cultivation (MSC) [30] gained recent attention due to its impressive practical performance, achieving a logical error rate of  $2 \times 10^{-9}$ at  $10^{-3}$  physical error rate. MSC injects a physical magic state into a surface code logical qubit, and suppresses infidelity by applying multiple rounds of post-selection and physical non-Clifford gates. Due to its use of exponentially scaling post-selection, MSC does not scale asymptotically and may not suffice for large scale FTQC.

In this work, we introduce new designs of magic state factories that integrate MSC on surface code and MSD in BB codes, achieving low logical error rates while using significantly fewer resources compared to conventional distillation factories. In more detail:

• We present a collection of MSD protocols that run

![](_page_1_Figure_0.jpeg)

Fig. 1: Magic-state factory design space. Target logical error rate as a function of available physical qubits for surface-code cultivation, BB-code distillation on gross and two-gross codes, and two-level protocols combining cultivation with two-gross or surface-code-only distillation at physical error rate  $p_{\rm phys}=10^{-3}$ . Our BB-based factories achieve lower output error at similar qubit budgets, and the cultivation + two-gross pipeline extends to lower error regimes than cultivation can reach.

entirely within a single BB code block<sup>1</sup>, in contrast to conventional factories which run MSD on multiple blocks of QLDPC codes or surface codes. This one block design confers significant savings in the space overhead of magic state factories, reducing the physical qubit count from thousands to hundreds. Additionally, it simplifies the architecture, decreases the number of long range interblock connections, and reduces decoding complexity.

• We present comprehensive, end-to-end optimizations for existing MSD circuits, including (i) logical qubit mapping to maximize the use of native, low cost logical gates<sup>2</sup> on the BB code, (ii) a masking technique that augments rotations with Z on inactive qubits to turn a set of sparse native logical operation on all logical qubits into a denser set over the active logical qubits, (iii) gate scheduling cast as a Traveling Salesman Problem to minimize compiled circuit depth, and (iv) parallel execution of two MSD protocols on the same BB code block.

Together, these methods significantly reduce the circuit depth of MSD and improve the factory throughput.

• We introduce a general method to compress MSD circuits to be supported on fewer logical qubits while maintaining the same level of logical error suppression. As examples, we compress the 49-to-1 protocol from 13 to 7 qubits, the 51-to-3CS protocol from 18 to 9 qubits, the 64-to-2CCZ protocol from 17 to 10 qubits, making these circuits implementable within a single BB code block. Moreover, this method is generic to any magic state distillation protocol generated by tri-orthogonal matrices. As near-term quantum computers are space-limited, our method

brings large MSD protocols significantly closer to practice.

We benchmark a collection of magic state factories

 We benchmark a collection of magic state factories with varying protocols, codes, and noise levels. Our results quantify the substantial overhead reduction enabled by the proposed optimizations, and, through sensitivity analysis, provide practical guidance for protocol design under different operating regimes, while highlighting the key bottlenecks and improvement opportunities for each design.

As shown in Figure 1, our leading proposal is to perform 15-to-1 MSD on a block of two-gross code (an instance of BB code), using logical magic states supplied by MSC on a surface code. Under  $10^{-3}$  physical error rate, this protocol is estimated to reach logical error rates lower than what is currently achievable through MSC, while using just one block of surface code and one block of two-gross code. Our factories naturally fit into the bicycle architecture [27], a promising near-term FTQC architecture based on the BB codes [16]. Conceptually, the bicycle architecture encodes information in BB codes and performs logical operations using recentlydeveloped code surgery techniques [31]–[34]. These operations consume magic states, which makes magic generation the central component that determines the overall efficiency. Our designs thereby constitute an essential improvement to the bicycle architecture. For general extractor architectures [26], our techniques can also be applied to design other efficient magic state factories.

The rest of this paper is organized as follows. Sec. II reviews the fundamentals of quantum error correction, the BB code family, and magic state distillation. In Sec. III, we present a full stack BB code based magic state distillation factory design. Sec. IV introduces multiple optimization methods to reduce both the depth and the size of the MSD circuit. Sec. V introduces a protocol-level compression method to reduce the logical-qubit footprint of the MSD circuit by recycling inactive logical qubits. In Sec. VI and Sec. VII, we evaluate the space time cost and output logical error rates of our approach and compare them with surface code based magic state distillation and cultivation.

#### II. BACKGROUND

