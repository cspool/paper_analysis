# IX. FUTURE WORK ON ATX PROGRAMMING

ATX opens up future directions both in computer architecture and in compilers and programming languages for accelerators. Currently, we provide source-code level (C/C++) support for ATX programming with the help of a small library that defines the UTE configuration functions of Figure [12](#page-7-1) (e.g., UTE cfg num streams), as well as the ATX instructions (e.g., ATXV1V1) in inline assembly. In the future, the first step is to extend kernel libraries such as MKL [\[78\]](#page-15-8) with hand-optimized, ready-to-use ATX kernels for common use cases.

Automating ATX programming is the next step. In particular, a compiler could be used to automatically generate source-level ATX programs from source-level non-ATX programs. Such a compiler would detect acceleratable code segments, identify streams and dependencies, configure the UTE, partition loops into tasks, and invoke NCAs using ATX instructions. Recently, Large Language Models (LLMs) have shown promise for source-to-source transformations of CPU code [\[56\]](#page-14-34). They may also prove useful to perform non-ATX to ATX source code transformations.

NCAs may not be the optimal choice for all acceleratable code. Hence, another future direction is to build compilers that, starting from CPU source code possibly written in a domainspecific language, automatically partition it into NCA, ICA, and OCA segments for maximum performance.

# IX. FUTURE WORK ON ATX PROGRAMMING

ATX opens up future directions both in computer architecture and in compilers and programming languages for accelerators. Currently, we provide source-code level (C/C++) support for ATX programming with the help of a small library that defines the UTE configuration functions of Figure [12](#page-7-1) (e.g., UTE cfg num streams), as well as the ATX instructions (e.g., ATXV1V1) in inline assembly. In the future, the first step is to extend kernel libraries such as MKL [\[78\]](#page-15-8) with hand-optimized, ready-to-use ATX kernels for common use cases.

Automating ATX programming is the next step. In particular, a compiler could be used to automatically generate source-level ATX programs from source-level non-ATX programs. Such a compiler would detect acceleratable code segments, identify streams and dependencies, configure the UTE, partition loops into tasks, and invoke NCAs using ATX instructions. Recently, Large Language Models (LLMs) have shown promise for source-to-source transformations of CPU code [\[56\]](#page-14-34). They may also prove useful to perform non-ATX to ATX source code transformations.

NCAs may not be the optimal choice for all acceleratable code. Hence, another future direction is to build compilers that, starting from CPU source code possibly written in a domainspecific language, automatically partition it into NCA, ICA, and OCA segments for maximum performance.

