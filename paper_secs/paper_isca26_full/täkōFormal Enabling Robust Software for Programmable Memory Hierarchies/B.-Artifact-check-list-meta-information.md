# *B. Artifact check-list (meta-information)*

- Run-time environment: Tested on Windows 11 Enterprise 23H2, with Dafny 4.11.0 and Alloy 6.2.0. Dafny on Windows requires .NET installation. Alloy requires Java (JVM 17+, tested with JVM 24.0.1).
- Hardware: Tested on 12th Gen Intel(R) Core i7-1280P Windows Laptop with 32 GB RAM.
- Metrics: Machine-Checked Proof Verification; confirming Litmus Test outcomes for Axiomatic Model.
- Output: Scripts will output results to the console. Dafny proof expected result: verification script succeeds. Alloy model expected result: Alloy execution matches expected result printed by script.
- Experiments: Experiment setup and listing included in README file.
- How much disk space required (approximately)?: With a Dafny installation and Alloy Installation, the full repository size is ∼180 MB.
- How much time is needed to prepare workflow (approximately)?: Around 15 minutes. (installing Dafny + Alloy, Java and potentially .NET).
- How much time is needed to complete experiments (approximately)?: Around 1.5 hours.
- Publicly available?: Yes, on GitHub.
- Code licenses (if publicly available)?: MIT License (included in repository).
- Workflow automation framework used?: Bash Scripts.
- Archived (provide DOI)?: Version 1.1.1 uploaded to https://doi.org/10.5281/zenodo.19444275

