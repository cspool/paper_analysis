# *A. Uncovering Ground Truth Configuration*

Our goal is to capture the RNG output before any hardware or software modifies it. While some devices directly provide an API for the raw TRNG output, others require modification to the source code to access it. Apollo 4 family of MCUs statically link shared crypto libraries at compile time, which means that we cannot modify the source code directly to get the raw TRNG output. In this case, we manually reverse engineer the binary by single stepping via a debugger to find the instruction at which the TRNG hardware populates its output at a memory-mapped output register. Then we patch the binary to get this raw RNG output instead of the one provided by the library API.

