# 6 Security Analysis

TZ-LLM protects the confidentiality of LLM parameters from any attacker who compromise the REE OS, the REE applications, or other TAs in the TEE.

Preventing direct access attacks. If an attacker in the REE tries to access plaintext parameters in the secure memory, the TZASC hardware blocks such access attempts. A malicious TA also cannot access the parameters in secure memory as the TEE OS enforces address space isolation between TAs.

If the attacker attempts to read the parameters in flash, he/she will only get content encrypted with a model key. The model key in flash is encrypted with a hardware-protected TEE key. It can only be decrypted by the TEE OS. The TEE OS only allows the LLM TA to access the model key.

Preventing DMA attacks. The attacker may exploit the NPU or other untrusted devices to initiate malicious DMA requests targeting parameters in the secure memory.

For the NPU, the TEE driver enforces two key protections before granting access to secure memory. First, it configures the TZPC to prohibit REE access to MMIO interface of the NPU. Second, it ensures that no NPU job previously launched by the REE driver is still executing. Therefore, the DMA destination can only be a benign address set by the TEE driver, preventing parameter leakage.

For untrusted devices, whether secure or non-secure, the TZASC is configured to reject any access from them to the secure memory regions for LLM parameters.

Preventing Iago attacks. Attackers may attempt to compromise the LLM TA or TEE OS for model theft by exploiting the interface between the TEE and the REE for Iago attacks. TZ-LLM exposes four TEE-REE interfaces vulnerable to Iago attacks: secure memory scaling, NPU job scheduling, model loading, and CPU thread scheduling.

For secure memory scaling, the CMA may return arbitrary memory addresses to the TEE. TZ-LLM counters this by validating the contiguity of the returned address against the previously allocated memory ([§4.2\)](#page-7-0). For NPU job scheduling, the REE NPU driver may schedule unauthorized secure jobs, replay previously scheduled jobs, or reorder them. TZ-LLM counters this by validating the job before execution ([§4.3\)](#page-7-1). For model loading ([§3.2\)](#page-4-3), a malicious REE OS may return forged results. TZ-LLM counters this by verifying the returned content using checksums. For CPU thread scheduling, the REE scheduler may violate the required execution order of TA threads. TZ-LLM counters this by managing synchronization primitives in the TEE ([§3.2\)](#page-4-3), ensuring that TA thread follows the execution order enforced by these primitives.

Side-channel and physical attack considerations. Existing side-channel attacks on TrustZone [\[45,](#page-15-13) [65,](#page-15-14) [94\]](#page-17-6) are outside the scope of this paper and have known mitigations [\[34,](#page-14-23) [59\]](#page-15-15). TZ-LLM may introduce two other side channels. First, the parameter tensor sizes are exposed to the REE when the TA scales secure memory. Second, the execution time of secure NPU jobs is exposed to the REE driver when it schedules the jobs. These channels may reveal model structures, but not parameter values. To the best of our knowledge, there are no public reports of side-channel attacks successfully stealing on-device LLM parameters. Additionally, these channels could be mitigated through orthogonal techniques such as dummy parameter loading and dummy computation.

Physical attacks through offline DRAM analysis, such as cold-boot attacks [\[92\]](#page-17-7), stem from TrustZone's hardware limitations. They can be mitigated by future memory encryption hardware [\[10\]](#page-14-19) and are orthogonal to TZ-LLM.

Ensuring the security of the existing TEE system. TZ-LLM minimizes its security impact on the existing TAs and TEE OS. First, TZ-LLM minimizes the modification of the privileged TEE OS to about 100 LoC ([§5\)](#page-9-0), by running the TEE NPU driver in user space, and designing simple secure memory scaling interfaces ([§4.2\)](#page-7-0). Second, even if the LLM TA is compromised, it cannot access the memory of other TAs or the TEE OS with direct read/write or NPU DMA, because the TEE OS enforces address space isolation, and the TZASC configuration only allows the NPU to access the memory regions for NPU job execution contexts ([§4.3\)](#page-7-1).

