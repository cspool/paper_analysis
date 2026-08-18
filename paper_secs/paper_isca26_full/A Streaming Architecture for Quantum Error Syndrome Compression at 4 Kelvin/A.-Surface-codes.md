# *A. Surface codes*

Quantum error correction (QEC) protects quantum information by encoding it across multiple physical qubits, collectively forming a logical qubit. Surface codes, the most adopted QEC scheme—especially for superconducting qubit implementations—arrange physical data qubits in a twodimensional lattice. These data qubits are interspersed and entangled with two types of ancilla qubits, which, when

![](_page_2_Figure_0.jpeg)

Fig. 2. Surface code with distance d = 3: smaller circles are data qubits, larger circles are ancilla qubits. Highlighted data qubits indicate physical errors; highlighted ancilla qubits indicate a syndrome value of 1. Cases (a) and (b): A phase-flip error (Z) triggers neighboring X ancilla qubits. Case (c): A bit-flip error (X) triggers Z ancilla qubits. Case (d): An error chain triggers ancilla qubits at its boundaries. Case (e): A bit- and phase-flip error (Y ) triggers all four neighbors. Case (f): A measurement error triggers a single ancilla qubit.

measured, produce syndromes that identify new bit-flip (X) and phase-flip (Z) errors, or a combination of those (Y ).

Considering that error detection in surface codes relies on parity checks, ancilla qubits detect error boundaries rather than individual errors. For example, a single X or Z error on a data qubit affects the parity checks of two adjacent ancilla qubits and results in two nonzero syndromes, while a Y error triggers four. Conversely, an ancilla qubit coupled to an even number of erroneous data qubits reports a zero syndrome, as the effects of the errors cancel out. This phenomenon leads to the formation of error chains. Figure 2 illustrates these scenarios using surface codes of distance d = 3. The distance d denotes the number of data qubits along the lattice's edges and determines the code's error correction capability.

In addition to data errors, noise can affect ancilla qubit measurements, producing measurement errors. These errors are distinguished from data errors through temporal analysis, which adds a third dimension to the decoding graph (Figure 3). Data errors produce nonzero syndromes only in the round they occur, as their effect on ancilla measurements remains consistent in subsequent rounds. Measurement errors manifest as transient changes in ancilla readings, producing paired nonzero syndromes in consecutive rounds.

IcePack is optimized for surface codes and addresses both data and measurement errors. The approach can be extended to other codes with a repeating grid structure. We provide further discussion in Section VII.

# *A. Surface codes*

Quantum error correction (QEC) protects quantum information by encoding it across multiple physical qubits, collectively forming a logical qubit. Surface codes, the most adopted QEC scheme—especially for superconducting qubit implementations—arrange physical data qubits in a twodimensional lattice. These data qubits are interspersed and entangled with two types of ancilla qubits, which, when

![](_page_2_Figure_0.jpeg)

Fig. 2. Surface code with distance d = 3: smaller circles are data qubits, larger circles are ancilla qubits. Highlighted data qubits indicate physical errors; highlighted ancilla qubits indicate a syndrome value of 1. Cases (a) and (b): A phase-flip error (Z) triggers neighboring X ancilla qubits. Case (c): A bit-flip error (X) triggers Z ancilla qubits. Case (d): An error chain triggers ancilla qubits at its boundaries. Case (e): A bit- and phase-flip error (Y ) triggers all four neighbors. Case (f): A measurement error triggers a single ancilla qubit.

measured, produce syndromes that identify new bit-flip (X) and phase-flip (Z) errors, or a combination of those (Y ).

Considering that error detection in surface codes relies on parity checks, ancilla qubits detect error boundaries rather than individual errors. For example, a single X or Z error on a data qubit affects the parity checks of two adjacent ancilla qubits and results in two nonzero syndromes, while a Y error triggers four. Conversely, an ancilla qubit coupled to an even number of erroneous data qubits reports a zero syndrome, as the effects of the errors cancel out. This phenomenon leads to the formation of error chains. Figure 2 illustrates these scenarios using surface codes of distance d = 3. The distance d denotes the number of data qubits along the lattice's edges and determines the code's error correction capability.

In addition to data errors, noise can affect ancilla qubit measurements, producing measurement errors. These errors are distinguished from data errors through temporal analysis, which adds a third dimension to the decoding graph (Figure 3). Data errors produce nonzero syndromes only in the round they occur, as their effect on ancilla measurements remains consistent in subsequent rounds. Measurement errors manifest as transient changes in ancilla readings, producing paired nonzero syndromes in consecutive rounds.

IcePack is optimized for surface codes and addresses both data and measurement errors. The approach can be extended to other codes with a repeating grid structure. We provide further discussion in Section VII.

