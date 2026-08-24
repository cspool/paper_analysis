# <span id="page-17-0"></span>B More Function Call Examples

Under the hood, models differ in how they surface tool calls in their chat templates and generations. For instance, Llama-3 variants may emit a function-style string func\_name(

param\_1=val\_1, param\_2=val\_2, ...), whereas Qwen-3 variants use "name": "func\_name", "arguments": .... Regardless of format, serving engines (e.g., vLLM, SGLang) include model-specific, template-aware parsers that take in the generated long string, recover the function name and parameters, and normalize them into the OpenAI-style schema, enabling uniform downstream handling. Thus, if we are using the general function calling interface provided by the serving engines, we don't need to worry about model-specific parsing.

For other use cases where the application is not using the function calling interface, and instead ask the model to output structured bash command via the chat interface, it's also easy to parse out the function name and arguments.

For example, in SWE Bench, to extract the intended tool invocation, just locate the single bash code block, split the command string on && or ||, then parse each sub-command: the first token is the executable/function name (pytest, git, . . . ) and the rest are its arguments.

```
1 pytest -q && git add -A && git commit -m
      " fix : handle None case in parser "
```

In Terminal Bench, this is even easier, as their structured format already handles the command splitting for us.

```
1 {
2 " state_analysis ": " The tests are
     failing with a NameError ." ,
3 " explanation ": " Open the file , fix the
      missing import and rerun tests ." ,
4 " commands ": [
5 { " keystrokes ": "vim src / app / main . py
     \n", " is_blocking ": false , "
     timeout_sec ": 2.0 },
6 { " keystrokes ": "pytest -q\n",
        " is_blocking ": true , "
     timeout_sec ": 30.0 }
7 ],
8 " is_task_complete ": false
9 }
```

