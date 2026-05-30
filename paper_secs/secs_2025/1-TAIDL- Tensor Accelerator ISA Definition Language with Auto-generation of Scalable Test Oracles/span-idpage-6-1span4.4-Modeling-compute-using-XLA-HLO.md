# <span id="page-6-1"></span>4.4 Modeling compute using XLA-HLO

One of the key design goals of TAIDL is to express the intent of the instructions without delving into the microarchitectural details. We observe that the computations done by an instruction of a tensor accelerator are primarily multi-dimensional tensor computations over the tensor buffers. Thus, the requirements for the compute syntax are that it should provide a high-level description, abstract the implementation details of the accelerator, and be flexible enough to define any fixed-shape multi-dimensional tensor computation. These requirements are satisfied by the rich operator set present in XLA-HLO IR used in the XLA compiler.

Revisiting Figure [2,](#page-2-0) the TAIDL definition for an AMX instruction uses operators bitcast-convert[3](#page-0-0) & reshape to load and store tile registers, operatorsreshape & transpose for layout transformation, and operators dot-general & convert for precise matrix multiplication.

<span id="page-6-3"></span>![](_page_6_Figure_9.jpeg)

Figure 9: TAIDL definition for switching data layouts from HWC\_C4 to CHW\_W4 (visualized for C = 8, H = 64, W = 16).

Figure [9](#page-6-3) shows the TAIDL definition of an instruction that switches data layout from channel-last (HWC\_C4) to row-major (CHW\_W4). It extensively uses XLA-HLO operators to convert the data layout into the original matrix and back to the target data layout. This instruction can be implemented using the Memory Layout Unit (MLU) in MTIA chips [\[24,](#page-14-24) [49\]](#page-15-29) or the BIRRD network in FEATHER [\[114\]](#page-17-1). This is a prime example where we have described the meaning of the instruction without diving into its implementation details.

