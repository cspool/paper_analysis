# **2 Background**

In this section, we introduce the necessary background to understand the proposed optimization on quantum Hamiltonian simulation. For basic quantum computing concepts (e.g., qubit, gate, linear operator, circuit), we recommend [\[36\]](#page-13-8) for more details.

### **2.1 Hamiltonian Simulation, Pauli Strings, and Trotterization**

The time evolution of a quantum system with its Hamiltonian *H* is characterized by the operator *e iHt* where *t* ∈ **R** representing the time. In general, directly translating the *e iHt* into a quantum algorithm is hard and a principled approaches are required.

In this passage we introduce the concept of Pauli string and Hamiltonian decomposition. In an n-qubit system, a Pauli string is defined as a length-n tensor product of the operators  $\{X,Y,Z,I\}$ , where each operator acts on a specific qubit index. This direct mapping of Pauli strings to qubits naturally arises in many quantum Hamiltonians, making them a convenient basis for both theoretical analyses and practical implementations.

The time evolution of a Pauli string, P is  $e^{iPt}$  and it can be synthesized into a quantum circuit using a series of Pauli gates, CNOT gates, and a Z-rotation gate exactly. This process works straightforwardly when dealing with a single Pauli string; however, challenges emerge when the objective is to synthesize an exponential of a sum of Pauli strings,  $\exp(it\sum_i P_i)$ . In these cases, closed-form analytical decompositions generally do not exist, which motivates the use of approximation techniques to break down the weighted sum of Pauli Strings into implementable quantum gate sequences.

It is known that all Pauli strings of length n formulate a basis for the linear space of all the Hermitian operators over n-qubits, and Hamiltonians are Hermitian operators. So a Hamiltonian can always be decomposed into a weighted sum of Pauli strings  $H = \sum_i w_i P_i$  where  $w_i \in \mathbb{R}$ . For simplicity, we absorb the weight and the associated Pauli string into one Hamiltonian term and denote  $H = \sum_i H_i$  in the rest of this paper. To approximate the exponential of the sum of Hamiltonian terms, one commonly employs Trotterization. Formally, it is based on the Lie–Trotter formula [20]:

$$e^{t(H_i + H_j)} \approx \left( e^{\frac{t}{N}H_i} e^{\frac{t}{N}H_j} \right)^N, \qquad (1)$$

$$\left\| e^{t(H_i + H_j)} - \left( e^{\frac{t}{N}H_i} e^{\frac{t}{N}H_j} \right)^N \right\| \leq \frac{t^2}{2N} \left\| [H_i, H_j] \right\| + \mathcal{O}\left( \frac{t^3}{N^2} \right),$$

where N is the number of Trotter steps, and the error depends on the sum of commutators  $[H_i,H_j]$  mitigated linearly by the number of Trotter steps. By splitting a large sum into smaller components that can be individually exponentiated, Trotterization provides a systematic method for approximating time-evolution operators. Increasing the number of Trotter steps reduces the approximation error but also increases the overall circuit depth. This method has been implemented in many industry and academia-offered software development kits [22, 16, 25] as a standard approach for quantum Hamiltonian simulation.

#### 2.2 Randomized Compilation

Randomized compilation has recently gained considerable attention in the quantum computing community as a means to mitigate coherent errors in quantum circuits. By converting systematic error into stochastic error, randomized compilation can improve the robustness of quantum algorithm approximations by allowing for better asymptotic scaling on larger time simulations. Early theoretical frameworks for randomized compilation were first presented in [4, 48, 46, 47, 17, 18], illustrating how randomly selected gate layers can effectively reduce correlated noise processes. In product formulas, random compilation can be invoked by shuffling each Trotter step, which would then cause rapidly changing signals and

<span id="page-2-0"></span>![](_page_2_Figure_7.jpeg)

Figure 2: The four stages of the Monte Carlo search tree. 1. Selection of a node for expansion and evaluation. 2) Expansion: choosing a new action and state combination that has not been explored. 3) Simulation: Randomly traversing states and actions to a terminal state and evaluating the outcome. 4) Backpropagation: updating tree metadata on outcomes learned through simulation

evolutions to average out erroneous terms [6], to give better scaling. This work leverages the idea of randomization to shuffle the orderings of partially Trotterized terms (introduced later) to turn coherent error into stochastic error.

