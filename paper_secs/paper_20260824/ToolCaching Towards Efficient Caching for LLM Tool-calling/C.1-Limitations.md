# C.1 Limitations

Although ToolCaching demonstrates substantial performance gains in LLM tool-calling scenarios, it primarily targets INFORMA-TIONAL tool calls, where the result of a call does not alter the

system state. However, with the increasing adoption of the Model Context Protocol (MCP) [\[19\]](#page-8-35), LLMs are becoming capable of invoking COMMAND-type tools that perform state-changing or irreversible operations.

As discussed in [Section 4,](#page-3-0) applying traditional caching strategies to such calls may lead to harmful side effects. Therefore, future caching frameworks must explicitly account for the semantics and side-effect characteristics of COMMAND calls, adopting fundamentally different mechanisms from those used for INFORMATIONAL requests.

