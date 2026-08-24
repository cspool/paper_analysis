# G Prompts

#### <span id="page-21-0"></span>G.1 Task Prompt for the Agent to Solve a GSO Task

#### Performance Optimization Task Prompt

I've uploaded a python code repository in the directory workspace\_dir\_name. Consider the following test script showing an example usage of the repository:

<test\_script>

[[ SPECIFICATION TEST]]

</test\_script>

Can you help me implement the necessary changes to the repository so that the runtime of the <test\_script> is optimized? Basic guidelines:

- 1. Your task is to make changes to non-test files in the /workspace directory to improve the performance of the <test\_script>.
- 2. Make changes while ensuring the repository is functionally equivalent to the original.
- 3. Do not overoptimize for just the specific inputs in <test\_script>. Make general performance improvements for the usage scenario shown.
- 4. You may need to rebuild the repo for your changes to take effect before testing. Some rebuilds may take time to run, so be patient with running them.

Follow these steps to improve performance:

- 1. As a first step, explore the repository structure.
- 2. Create a script in the /workspace directory (e.g., /workspace/test\_opt.py) to reproduce and time the example, then execute it with python /workspace/<filename.py>.
- 3. Edit the source code of the repository to improve performance.
- 4. Rebuild and rerun your script to confirm that performance has improved.

