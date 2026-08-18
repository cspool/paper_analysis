# *D. Major Claims*

- *1) C1:* Context switches reliably clear the x18 register. Based on this property, we build TIDE, a new technique for detecting interrupts without relying on any architectural timer.
- *2) C2:* Unlike Linux-based systems, Apple Silicon running macOS does not employ interrupt affinity for shared peripheral interrupts. Instead, these interrupts are distributed uniformly across active cores, while idle cores are excluded.
- *3) C3:* TIDE enables realistic interrupt-based side-channel attacks without requiring explicit core binding.

### *E. Experiments*

### *1) E1:* Validating the x18 behavior

How to. We use a timestamp-based interrupt detection method as a baseline to identify interrupt arrivals. During this process, we repeatedly set the x18 register to a nonzero value and check whether it is cleared when an interrupt is detected by the baseline method.

Results. The artifact shows that the x18 register is cleared in 100% of detected interrupts. This result validates the key assumption underlying TIDE: it can precisely detect interrupts without requiring a high-resolution timer.

*2) E2:* Reverse-engineering the Apple interrupt controller How to. We generate controlled network interrupts and detect them using TIDE. We then compute the ratio of generated interrupt requests to successfully observed interrupts, referred to as the efficiency metric in Section [IV.](#page-6-3)

Results. The artifact reproduces the behavior reported in Figure [6](#page-8-2) of the paper. In contrast to Linux-based systems, Apple Silicon running macOS does not deliver shared peripheral interrupts to a fixed core. Instead, shared peripheral interrupts are distributed uniformly across active cores, while idle cores do not receive them.

*3) E3:* Applying TIDE to a realistic attack

How to. We apply TIDE to monitor website-induced interrupts while a webpage is loading. The resulting traces are then used to infer the visited website.

Results. The artifact reproduces the end-to-end website fingerprinting attack reported in Table [III](#page-9-1) of the paper. Even without explicit core binding, TIDE infers website visits with high accuracy, achieving 93.8% top-1 accuracy on Safari on our test machine.

For additional details on how to build, run, and validate each experiment, please refer to the README.md file in the corresponding subdirectory.