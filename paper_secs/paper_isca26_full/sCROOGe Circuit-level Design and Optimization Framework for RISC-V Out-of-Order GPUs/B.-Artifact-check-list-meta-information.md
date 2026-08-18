# *B. Artifact check-list (meta-information)*

- Program: The artifact includes the System Verilog source code of the OoO schemes implementation as well as the source code of the benchmarks listed in VII and seen in Fig. 17.
- Compilation: All needed tools are included in the container and have the appropriate versions.
- Transformations: The required execution scheme is passed as a bash script option to /vortex/ci/blackbox.sh.
- Binary: The artifact does not contain pre-compiled binaries. Everything is compiled through the bash scripts in /vortex.
- Data set: Post-synthesis as well as post Place-and-Route power and area measurements are included across technology nodes.
- Run-time environment: A Docker container image is provided to run on a linux machine. All the required tools as well as python package dependencies are installed within the image.
- Hardware: A machine with > 32GB RAM.
- Execution: The performance evaluation experiments can take up to ∼ 270h to complete (single process).
- Metrics: Per-kernel total instructions and cycles are generated through RTL simulation using Verilator.
- Output: Validation plots for Fig. 1,6,9-24 as well as all intermediate .csv files.
- Experiments: The artifact includes a script that downloads the appropriate container image. All necessary source code is found within the container, along with pre-computed dynamic instruction traces and intermediate .csv files. RTL simulation experiments using Verilator overwrite these files with results obtained anew and validation plots can thus be produced.
- How much time is needed to prepare workflow (approximately)?: < 5 minutes.
- How much time is needed to complete experiments (approximately)?: ∼ 100h for Fig. 14, an additional ∼ 170h (single process) for the rest of the results.
- Publicly available?: Yes.

