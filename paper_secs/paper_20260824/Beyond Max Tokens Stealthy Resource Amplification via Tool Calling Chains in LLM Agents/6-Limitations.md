# 6 Limitations

Our evaluation is conducted in a controlled agent emulator with deterministic, stubbed tool payloads, and focuses on single-tool, single-turn subsets of ToolBench/BFCL for comparability; results may differ in production agent stacks, multitool long-horizon tasks, and under different runtimes/hardware. Moreover, while our template instantiation uses a structured numeric calibration sequence to reliably elicit long tool-calling outputs, similar resource amplification could be realized through other protocol-compliant long-form content patterns beyond number lists, which we do not exhaustively explore.

