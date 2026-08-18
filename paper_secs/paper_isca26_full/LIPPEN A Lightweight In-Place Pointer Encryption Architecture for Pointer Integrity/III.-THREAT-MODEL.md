# III. THREAT MODEL

We follow a threat model similar to typical memorycorruption attacks [67]. The attacker aims to exploit memorysafety vulnerabilities, such as stack buffer overflows and use-after-free bugs, to corrupt pointers and thereby subvert control flow or mount data-oriented attacks. We assume a powerful attacker who, after exploiting such a vulnerability, can perform arbitrary reads of process memory and overwrite any writable memory location, including code pointers, data pointers, and user-space protection metadata such as modifiers. This capability enables both control-flow hijacking and dataoriented manipulation.

We also include pointer forgery during transient execution attacks [42], [54], [70] in our threat model. If pointer integrity is violated during speculation (e.g., Speculative ROP [54], or forging a pointer in Spectre attacks), we consider it an attack. However, we focus on pointer integrity protection; other information leakages due to side channels not using pointer forgery are out of scope.

We trust the underlying hardware and operating system kernel, which are responsible for securely generating, managing, and storing the process-wide secret key K, which remains constant during execution and is inaccessible to user-level code.

## IV. LIPPEN DESIGN

