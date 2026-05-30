# <span id="page-6-0"></span>4.3 Tensor Buffers vs. Control Registers

Tensor buffers and control registers represent two different types of data stored on an accelerator. Control register values are independent of the input tensor data for a computation graph. Therefore, they can be statically analyzed at the time of simulation. However, the computation performed by an instruction is determined by the control registers, like configuration flags. Therefore, the data stored in tensor buffers depends on both the control registers and input tensor data. This characteristic means that an accelerator contains two types of data – one that can be statically analyzed at simulation-time and one that is dependent on the input tensor data.

Revisiting the TPUv1 data model in Figure [4,](#page-4-2) the data present in the unified buffer, accumulator buffer, systolic array weight matrix, and FIFO weights buffer (lines 2 to 5) varies based on the program data. Whereas the values of occupancy, push, and pop registers (lines 7 to 9) are constant for a particular code segment.

