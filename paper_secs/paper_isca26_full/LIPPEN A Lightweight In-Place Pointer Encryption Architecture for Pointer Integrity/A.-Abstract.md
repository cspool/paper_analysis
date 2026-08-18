# *A. Abstract*

The artifact is designed to enable reproduction of the core components and a representative subset of the evaluation results, while supporting full reproduction given additional hardware and setup time.

The artifact guides users through compiling and simulating the Chipyard design with Verilator and running small test programs. It also includes instructions for generating a bitstream for the FPGA implementation. To run the microbenchmarks on our prototype, users need access to a Xilinx VCU118 FPGA and must follow the documented steps for building the Linux image, preparing the SD card with the required RISC-V binaries, and transferring files through the SD card workflow. In addition, the artifact includes microbenchmarks for ARM64 processors, which we tested on Apple M1 systems and expect to work on other Apple M-series processors as well. Because Apple restricts the use of PAC instructions in userspace programs, users must disable System Integrity Protection before running those experiments.

## *B. Artifact check-list (meta-information)*

- Compilation: Compiling the modified Chipyard hardware design, the LLVM-based compiler toolchain, and the provided microbenchmarks.
- Run-time environment: Ubuntu Linux for building and simulation; firemarshal linux for runnign on FPGA; macOS for ARM64 microbenchmark experiments.
- Hardware: A server or workstation for building and simulation; a Xilinx VCU118 FPGA for FPGA-based experiments; an Apple M1-based machine for ARM64 microbenchmark evaluation.
- Execution: Running Verilator-based simulations, generating FPGA bitstreams, booting Linux on the FPGA prototype, and executing the provided microbenchmarks both on FPGA and Apple M1.
- Output: Generated simulation binaries, FPGA bitstreams, compiled benchmark binaries, and performance measurement results.
- How much time is needed to prepare workflow (approximately)?: Around 30–60 minutes for software and simulation setup; the FPGA synthesis and bitstream generation take hours.
- Publicly available?: Yes, artifacts can be found here: https://doi.org/10.5281/zenodo.19901476 https://github.com/bearhw/LIPPEN
- Code licenses (if publicly available)?: GNU General Public License v3.0.

