# <span id="page-15-1"></span>A Supplemental Figures

Figure 15: Ensemble Struct Example.

<span id="page-15-0"></span>> **[图片提取文字 (无描述)]:**
> Indexer Summary Chunk 1 **Building Block** Code Code Summary Chunk 2 Chunk N **XDB** Similarity Search Retriever User Question Building Building Block 1 Block 1 Building Building Block 2 Block 2 Selector Building .... Block 3 Building Block 10
![](_page_15_Figure_5.jpeg)

Figure 16: RAG to Assist Planning

```
 Pydantic Example:
class QueryInput(BaseModel):
 text: str = Field(...,
 description="user input request")
class QueryOutput(BaseModel):
 entity: ODSEntity = Field(name="device")
 key: ODSKey = Field(name="interface")
 transformation: List[str] = ["diff"]
 reduction: List[str] = ["p90","avg"]
 start_timestamp: int = -3600
 end_timestamp: int = 0
 Run Context Example:
 class AnalectRunContext(BaseModel):
  session: str = "s1"
  io: IOInterface = "UI"
  llm_manager: LLManager =
  LLMParams(model="llama3-70b")
                                                        class ODSQuerier(LLMAnalect[QueryInput, QueryOutput]):
                                                         def display_name(cls) -> str:
                                                         return "ODS Querier"
                                                         def input_examples(cls) -> list[EntryInput]:
                                                         return [EntryInput(
                                                         question="Show me p90 CPU utilization 
                                                         for all routers in X"),
                                                         EntryInput(
                                                         question="What is average egress link
                                                         utilization in X?"
                                                         )]
                                                         async def impl(self, inp: QueryInput,
                                                         context: AnalectRunContext)
                                                         -> QueryOutput:
                                                         ...
                                                         Analect Example:
```

Figure 17: An example of Pydantic, Analect, and Run Context.

<span id="page-15-2"></span>> **[图片提取文字 (无描述)]:**
> CollectorTask description: "You are a code generating agent for TML queries. Your task is to create the corresponding TML query for the user's question above. TML, which stands for Topology Modification Language, is a Python-based DSL that..." examples: [Example(·)] CollectorThinking CollectorExample user\_input: "How to update content: "User wants to update L3 nodes, this is a spectrum for my given set of tm.update type query. User mentions a known set fibers?" of fibers already. Create a dict assistant\_response: [Thinking(·), - - ▶ 'fiber\_spectrum\_overrides' with fibers and their desired spectrum. Inject this into the TML update." Artifact(1)1 CollectorArtifact content: fiber spectrum overrides = { "<fiber 1>": <spectrum size ghz per fp value>, "<fiber 2>": <spectrum size ghz per fp value>, tm.update( what="fibers", update="ownedFiberProperties= ownedFiberProperties(spectrum size ghz per fp=fiber spectrum overrides[name])", where="name in fiber spectrum overrides", fiber spectrum overrides=fiber spectrum overrides,
![](_page_15_Figure_9.jpeg)

Figure 18: Example of Collector for TML use case.