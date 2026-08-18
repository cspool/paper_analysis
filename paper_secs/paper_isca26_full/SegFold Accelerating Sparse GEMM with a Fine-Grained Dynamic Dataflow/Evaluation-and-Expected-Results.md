# *Evaluation and Expected Results*

Because the simulator is fully deterministic, the CSV outputs should be bit-identical to the reference files shipped in expected\_results/. The generated plots faithfully reproduce Figures 8–12 of the paper.

In addition, the hardware/ directory ships SystemVerilog RTL for every SegFold module together with synthesis reports (area, power, and timing) produced by Synopsys Design Compiler targeting the ASAP 7nm standard cell library. These correspond directly to the hardware cost analysis in the paper.

