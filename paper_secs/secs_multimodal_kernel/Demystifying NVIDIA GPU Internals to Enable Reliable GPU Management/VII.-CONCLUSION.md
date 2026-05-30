# VII. CONCLUSION

<span id="page-9-0"></span>In this work, we identified key behaviors of NVIDIA GPUs that can compromise the safety of preexisting real-time GPU management frameworks. In order to protect future GPUmanagement efforts from such hazards, we experimentally derived and codified GPU scheduling rules for the lowest-level scheduling primitives—channels, runlists, and engines. We analytically evaluate the necessity of our rules for GPU management and analysis, finding that our rules expose counterexamples in three prior works. To assist others in maintaining and adding to our scheduling rules, we open-sourced all our tools, including our nvdebug Linux kernel module.

In future work, we aim to develop rules for how runlists are arbitrated and budgeted when containing multiple tasks; a sufficiently-high-accuracy model may allow offloading scheduling from management frameworks onto the HW scheduler—reducing capacity loss.

