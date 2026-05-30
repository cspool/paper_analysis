# <span id="page-8-2"></span>4.9 Modeling Warp Instruction Distribution

GCoM improves the model accuracy by considering the workload imbalance caused by the warp number distribution in the entire GPU. But we find that in LLM inference, the workload imbalance does not appear in warp number distribution. Instead, it appears in warp instruction number distribution. For accurately modeling kernel cycles in LLM inference, modeling instruction divergence () is important. We employ equation [\(29\)](#page-8-6) to model .

<span id="page-8-6"></span>
$$ID = (\text{maxsub-coreInstr} - I_{SC\_Repr.warp}) / IssueRate$$
 (29)

where the − is the maximum number of warp instructions on a sub-core in the kernel. \_ . is the number of warp instructions of the sub-core which executes the representative warp. is the instruction issuing rate in instructions per cycle. As such, the unit of is cycles. By adding it to other stalled cycles shown in equation [\(16\)](#page-6-2), we can improve the accuracy of modeling kernel cycles in LLM inference.

