# D. Overall Architecture

Fig. 13 illustrates the overall architecture of Anda, which includes the top controller, address generator, activation buffer, weight buffer, matrix computation unit (MXU), vector unit, and bit-plane compressor. The LLM inference is orchestrated as follows: • Initially, the instruction memory is programmed through the I/O interface of the top controller, which governs the address generator during operation. • The address generator produces read and write addresses for both activation and weight buffers. Both the activation buffer and weight buffer follow the proposed bit-plane-based data layout for efficient data handling. • The MXU, featuring a  $16 \times 16$  APU array, performs FP-INT GeMM operations following typical output stationary dataflow [45]. The weight data dispatcher, equipped with registers, allows overlapping weight loading and computation, broadcasting weights row-wise to each APU

![](_page_7_Figure_6.jpeg)

Fig. 13. Anda system architecture.

for data reuse. The activation data dispatcher supplies a bitplane vector of activations each cycle, sequentially feeding it into the MXU and sharing it across columns to maximize input reuse and enable multiple calculations with the same input. Upon completing the GeMM computation, the output results are delivered to the BPC via the output data dispatcher. Complementing MXU, the vector unit processes the nonlinear functions of the transformer block. FP16 outputs of MXU or vector unit can be optionally compressed to Anda format by the BPC, optimizing storage efficiency. Processed outputs are written back to the activation buffer. Finally, activation results are transferred to external memory for subsequent operations.

#### V. EVALUATION

