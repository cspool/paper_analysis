# Prompt H.1: Template for the HTML to TSV Task

#### [TASK]

```
Your task is to extract specific information from an HTML webpage and output the extracted
information in a tsv file. You will be first given an HTML webpage. Then, you should follow the
specific instruction provided later and output the tsv file following the format provided in the
instruction.
[INPUT WEBPAGE]
"'html
html page
"'
[TARGET INFORMATION]
Based on the HTML webpage above about {task topic}, extract the following properties from the
items listed on the webpage: target information
[OUTPUT FORMAT]
Structure your output in TSV format such that each row of your output corresponds to the
aforementioned properties of an item and each property is separated from each other by a tab " ".
Your output should be in the following format:
"'tsv
tsv header
{{Your TSV output}}
"'
```

### [IMPORTANT NOTES]

- Make sure that you have read through all items listed on the webpage and followed the same order as they appear on the webpage.
- If you are asked to only extract some rows that satisfy specific conditions, ONLY extract those rows that satisfy the conditions and do NOT include other irrelevant rows in your output.
- If a property of an item is blank, not applicable, or not parseable, please set the property to "N/A" for the item.
- If a property spans multiple lines, please extract all the lines and replace the newline character with a space character.
- If a property consists of a list of items, please replace the newline character with a space character and separate the items with a comma ",".
- If there are any special characters, numerical values of a specific format, or any unusual formatting in the property, please keep them as they are. If the property comes with a unit, please keep the unit as well in the property.
- Do not include html tags in the extracted information. Only include the text.
- Do not provide any additional information in your output other than the tsv.

Now, extract the information from the HTML webpage above and follow the output format above in your answer.

### <span id="page-43-0"></span>Prompt H.2: Template for the Pseudocode to Code Task

#### [TASK]:

You will be given lines of pseudocode, your task is to write the corresponding C++ code. The pseudocode will provide detailed description of the c++ code line by line. The pseudocode is garanteed to be correct and complete.

