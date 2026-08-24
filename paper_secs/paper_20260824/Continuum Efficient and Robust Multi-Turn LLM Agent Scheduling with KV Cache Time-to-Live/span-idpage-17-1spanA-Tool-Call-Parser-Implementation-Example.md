# <span id="page-17-1"></span>A Tool Call Parser Implementation Example

We attach the implementation for the tool parser for mini-SWE-agent here.

```
1 class ToolCallParser :
2 """Parser for extracting function calls from
     LLM output.
4 Uses the same parsing logic as mini -swe -agent
     to extract bash commands
5 from markdown code blocks and identify the
     function call.
7 This can be extended for other datasets with
     different parsing logic.
8 """
10 def parse ( self , text : str) -> Optional [str]:
11 """Parse LLM output and extract the
     function call name.
13 Args:
14 text: Output text from the LLM
16 Returns:
17 The function call name (e.g., "ls", "
     cd", "git"), or None if not found
18 """
19 # Same regex pattern as mini -swe -agent: r
     "'''bash\s*\n(.*?)\n' ' '"
20 actions = re . findall (r"'''bash\s*\n(.*?)\n
     '''", text , re . DOTALL )
22 if len( actions ) == 1:
23 bash_action = actions [0]. strip ()
24 # Extract the first word (command)
     from the action
25 words = bash_action . split ()
26 if words :
27 return words [0]
29 return None
```

Listing 1: Tool Call Parser Example

