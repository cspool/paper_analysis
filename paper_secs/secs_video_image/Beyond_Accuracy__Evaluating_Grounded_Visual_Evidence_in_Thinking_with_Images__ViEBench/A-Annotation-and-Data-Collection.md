# A Annotation and Data Collection

To ensure high-quality grounding and reasoning pairs, we developed a specialized web-based annotation platform based on Label Studio [\(Tkachenko](#page-9-20) [et al.,](#page-9-20) [2020-2025\)](#page-9-20) as shown in Fig. [A1.](#page-11-0) The interface is designed to facilitate the synchronized collection of visual bounding boxes, natural language queries, and logical categorizations.

The annotation pipeline consists of the following four key modules:

- Interactive Image Canvas: The core of the interface allows annotators to perform multilevel inspection. Supporting standard zooming and panning operations, it enables annotators to locate minuscule targets that require agentic "Thinking-with-Images" behavior.
- Bounding Box Grounding: Annotators are required to draw a precise gold bounding box (Bgt) around the visual evidence necessary to answer the question. This provides the ground-truth for calculating the Intersectionover-Area (IoA) metrics used in our evaluation.
- Q&A Annotation: Two dedicated text fields are provided for annotators to author the Question and the corresponding Answer. Annotators are instructed to ensure that the question cannot be answered confidently without referring to the fine-grained details within the specified bounding box.
- Task Categorization: Each sample is manually classified into one of two categories:
  - 1. Perception: Questions focusing on direct attribute recognition or simple object identification.
  - 2. Reasoning: Questions requiring multistep logical deduction, spatial relationship analysis, or the synthesis of internal knowledge with visual evidence.

## B Case Study

In our case studies (Fig. [A2\)](#page-12-0), Pixel Reasoner [\(Su et al.,](#page-9-0) [2025a\)](#page-9-0) demonstrates a representative instance of Ground-Success & Answer-Failure (G<sup>+</sup> · A−). When tasked with a query requiring specific posture recognition, the model accurately identifies the need to focus on the "child at the bottom right" and generates a high-precision crop that perfectly encompasses the gold BBox. However,

<span id="page-11-0"></span>![](_page_11_Picture_11.jpeg)

Figure A1: The web-based annotation interface used for ViEBench data collection. It supports interactive bounding box drawing, Q&A entry, and fine-grained category selection.

it still misinterprets the child's posture as "sitting" rather than "standing." This instance serves as evidence that successful visual localization does not inherently guarantee logical understanding. It highlights a cognitive bottleneck where models struggle to synthesize fine-grained visual semantics into a correct reasoning chain, even when the relevant pixels are clearly in view.

Conversely, Thyme [\(Zhang et al.,](#page-10-5) [2025b\)](#page-10-5) illustrates Ungrounded Correct Answer (G<sup>−</sup> · A+) behavior when tasked to determine the "color of the hat worn by the child in the yellow shirt." Although its CoT correctly identifies the intent to zoom into the specific region, the actual executed crop coordinates are significantly shifted toward an irrelevant background area, missing the gold BBox entirely. Despite this localization failure, the model arrives at the correct answer. This phenomenon suggests that the model may be utilizing broader global context cues rather than specific grounded evidence. Such cases underscore the necessity of process-

<span id="page-12-0"></span>![](_page_12_Figure_0.jpeg)

Figure A2: (Top) Ground-Success and Answer-Failure: Pixel Reasoner achieves near-perfect spatial grounding on the target child but fails to correctly interpret the child's physical state from the high-resolution crop, leading to an incorrect final answer. (Bottom) Ungrounded Correct Answer: Thyme arrives at the correct answer despite focusing on an irrelevant background region far from the gold BBox. This exposes a redundant cropping behavior, where the model's tool-invocation process is functionally decoupled from its final decision.

level auditing provided by ViEBench to distinguish between faithful visual operations and results that lack grounded evidence.