# *B. Execution Model*

From our preceding analysis, we can distill all the operating patterns into five fundamental, high-level behaviors that this model must support.

- 1) *Connectivity Configuration.* Activating or deactivating couplings to allocate resources in all processing schemes.
- 2) *Data Loading.* Setting initial node and coupling values.
- 3) *Component Clamping.* Locking the states of selected nodes or couplings.

- 4) *System Evolution Management.* Initiating and controlling the duration of the synchronous, parallel evolution of all non-locked components within scope.
- 5) *Results Retrieval.* Reading states from specified components, whether retrieving node values as a solution or reading coupling values as trained parameters.

Summarizing the behaviors, our complete execution model is defined by this load-lock-evolve-store flow acting upon a system topology defined with connection reconfigurability.

![](_page_4_Figure_12.jpeg)

Fig. 5. Execution flow of representative applications. The curved arrows form loops. Loop in ML Training: iteratively load new data. Loop in Optimization: modify data for annealing. Loop in DE Solving: updating boundary conditions.

Figure 5 shows how these representative applications are mapped to the execution model. For instance, ML Inference follows a simple, linear sequence: *Connect* couplings to define the model topology, *Load* nodes and couplings to set inputs and weights, *Lock* nodes and couplings to establish boundary conditions, a single node *Evolution* step for computation, and a final node *Store* to retrieve the result.

In contrast, other applications require iterative patterns. ML Training typically enters an *Evolve-Load* loop, where couplings evolve, and new training data is iteratively loaded before a final coupling store to save the trained weights. Similarly, Optimization may use an *Evolve-Store-Load* loop to modify data for a new annealing step, and DE Solving can use an *Evolve-Load* loop to update boundary conditions for time-dependent problems. This demonstrates that the loadlock-evolve-store model is not just a fixed sequence but a set of composable phases, flexible enough to describe linear, iterative, node-evolving, and coupling-evolving computations, thus validating it as a unified model for diverse DSU applications.

#### *C. Instruction Format*

To implement the execution model, we define a minimalist 9-instruction ISA, dubbed DS-ISA. As shown in Figure 6, these instructions are partitioned into three logical categories that directly map to the above execution model. Specifically, four instructions for the node lifecycle (e.g., N LOAD, N LOCK), four for the coupling (e.g., C LOAD, C LOCK), and an additional but critical instruction for connectivity configuration (CFG CONN).

|        | 63     |  | 56 55              |         | 24 23    |                                                       | 17 16  |  |        | 8 7      | 0 |
|--------|--------|--|--------------------|---------|----------|-------------------------------------------------------|--------|--|--------|----------|---|
| E-Type | Opcode |  | Imm_address        |         | Imm_time |                                                       |        |  |        | Reserved |   |
|        | 1 byte |  | 4 bytes            |         | 2 bytes  |                                                       |        |  |        | 1 byte   |   |
| N-Type |        |  | Opcode Imm_address |         | Imm_NGID |                                                       |        |  |        | Reserved |   |
|        | 1 byte |  |                    | 4 bytes |          | 2 bytes                                               |        |  |        | 1 byte   |   |
| C-Type |        |  |                    |         |          | Opcode Imm_address Imm_CGID_col Imm_CGID_row Reserved |        |  |        |          |   |
|        | 1 byte |  | 4 bytes            |         | 1 byte   |                                                       | 1 byte |  | 1 byte |          |   |

| Category<br>Name       |   |                     | DS-ISA                        |
|------------------------|---|---------------------|-------------------------------|
| Load Node Group        | N | N_LOAD              | Data Addr, NGID               |
| Lock Node Group        |   | N_LOCK              | NLM Addr, NGID                |
| Evolve Node Groups     |   | N_EVOLVE            | GM Addr, Time                 |
| Store Node Group       | N | N_STORE             | Data Addr, NGID               |
| Load Coupling Group    | C | C_LOAD              | Data Addr, CGID_col, CGID_row |
| Lock Coupling Group    | C | C_LOCK              | CLM Addr, CGID_col, CGID_row  |
| Evolve Coupling Groups |   | C_EVOLVE            | GM Addr, Time                 |
| Store Coupling Group   | C | C_STORE             | Data Addr, CGID_col, CGID_row |
| Configure Connection   | C | CFG_CONN            | CM Addr, NGID                 |
|                        |   | Type<br>N<br>E<br>E |                               |

Fig. 6. Instruction and format definitions of DS-ISA. NLM: Node Lock Mask; GM: Group Mask; CLM: Coupling Lock Mask; CM: Connection Mask; GID: Group ID; CGID: Coupling Group ID.

This ISA follows a unique "label-and-trigger" computing mechanism, which is essential for managing synchronous, collective evolution. The N LOCK and C LOCK instructions are labeling commands that set lock masks, while the EVOLVE instructions then act as the trigger, initiating a single, collective execution by applying the pre-set masks simultaneously. This separation ensures non-locked components to begin their parallel evolution from a synchronized state.

With the set of operations defined, we must design a format that addresses the core challenge of controlling a potentially vast number of nodes and couplings. We draw inspiration from GPUs and organize nodes and couplings into groups, such that a single instruction operates on all elements within a group synchronously in lockstep. This grouping strategy simplifies the primary control problem, and a two-level, hierarchical control scheme is adopted accordingly: First, we require a mechanism for inter-group control to select which groups participate in a collective action, such as the Group Mask (GM) used to manage parallel evolution. Second, we require intragroup control for fine-grained manipulation. This includes onedimensional Node Lock Masks (NLM) for setting boundary conditions, as well as two-dimensional coupling masks. These coupling masks, such as the Coupling Lock Mask (CLM) and Connection Mask (CM), are defined by their column and row mask components to provide fine-grained control over the selected coupling group, which is specified using column and row components in Coupling Group ID (CGID).

However, this mask-based control scheme has a direct and critical implication for the instruction format: scalability. The size of these intra-group masks (e.g., NLM) scales linearly with the size of the group, and the inter-group masks (e.g., GM) scale with the number of groups. It is therefore architecturally infeasible to embed this large, variable-sized data directly into an instruction.

To resolve this, we adopt an indirect control scheme. Rather than embedding the large, scalable masks into the instruction itself, the instruction carries an address to the data in memory (e.g., on-chip SRAM). Based on this consideration, we adopt a fixed-length 64-bit instruction format, providing ample space to hold both a large address pointer and other immediate control values. This format, detailed in Figure 6, is partitioned into three distinct types (E-Type, N-Type, and C-Type) based on its operands. The 4-byte Imm address field provides a 32 bit address, which points to data and scalable masks (NLM, CLM, CM, GM) in memory. The 2-byte immediate field, in contrast, is used for data that is small or scales logarithmically, such as the Imm NGID, the Imm CGID col/Imm CGID row components, or the evolution duration Imm time. This twolevel immediate system allows our 64-bit instruction to control DSUs of extensible scale by loading the appropriate masks from memory, providing a simple, scalable, and efficient ISA.

To make a concrete connection between application and DS-ISA, Figure 7 illustrates how the DS-ISA executes a simple ML inference task following the load-lock-evolve-store model. To achieve this, input features must be mapped to input node groups, output features to output node groups, with couplings encoding the model weights that drive the influence from input to output. The procedures are: (1) To determine which nodes serve as input and output nodes, a CFG CONN instruction uses an Intra-Group Mask to CONNECT the corresponding couplings representing the influence from input to output. In this example, the highlighted coupling groups suggest that the first four node groups influence the next six node groups. (2) N LOAD and C LOAD instructions use the Memory Interface to load input data and weights into these components. (3) as inputs and weights are locked while outputs are free to evolve, N LOCK and C LOCK apply their Intra-Group Masks to lock specific members in the input node groups and corresponding coupling groups. (4) N EVOLVE instruction then uses an Inter-Group Mask to Evolve only the output node group. (5) to write back, N STORE uses the Memory Interface to Store the resulting states from the evolved nodes. This example shows how the ISA's core mechanisms, such as intra-group and intergroup masking, provide the necessary control to execute the distinct phases of our model for a practical application.

# *B. Execution Model*

From our preceding analysis, we can distill all the operating patterns into five fundamental, high-level behaviors that this model must support.

- 1) *Connectivity Configuration.* Activating or deactivating couplings to allocate resources in all processing schemes.
- 2) *Data Loading.* Setting initial node and coupling values.
- 3) *Component Clamping.* Locking the states of selected nodes or couplings.

- 4) *System Evolution Management.* Initiating and controlling the duration of the synchronous, parallel evolution of all non-locked components within scope.
- 5) *Results Retrieval.* Reading states from specified components, whether retrieving node values as a solution or reading coupling values as trained parameters.

Summarizing the behaviors, our complete execution model is defined by this load-lock-evolve-store flow acting upon a system topology defined with connection reconfigurability.

![](_page_4_Figure_12.jpeg)

Fig. 5. Execution flow of representative applications. The curved arrows form loops. Loop in ML Training: iteratively load new data. Loop in Optimization: modify data for annealing. Loop in DE Solving: updating boundary conditions.

Figure 5 shows how these representative applications are mapped to the execution model. For instance, ML Inference follows a simple, linear sequence: *Connect* couplings to define the model topology, *Load* nodes and couplings to set inputs and weights, *Lock* nodes and couplings to establish boundary conditions, a single node *Evolution* step for computation, and a final node *Store* to retrieve the result.

In contrast, other applications require iterative patterns. ML Training typically enters an *Evolve-Load* loop, where couplings evolve, and new training data is iteratively loaded before a final coupling store to save the trained weights. Similarly, Optimization may use an *Evolve-Store-Load* loop to modify data for a new annealing step, and DE Solving can use an *Evolve-Load* loop to update boundary conditions for time-dependent problems. This demonstrates that the loadlock-evolve-store model is not just a fixed sequence but a set of composable phases, flexible enough to describe linear, iterative, node-evolving, and coupling-evolving computations, thus validating it as a unified model for diverse DSU applications.

#### *C. Instruction Format*

To implement the execution model, we define a minimalist 9-instruction ISA, dubbed DS-ISA. As shown in Figure 6, these instructions are partitioned into three logical categories that directly map to the above execution model. Specifically, four instructions for the node lifecycle (e.g., N LOAD, N LOCK), four for the coupling (e.g., C LOAD, C LOCK), and an additional but critical instruction for connectivity configuration (CFG CONN).

|        | 63     |  | 56 55              |         | 24 23    |                                                       | 17 16  |  |        | 8 7      | 0 |
|--------|--------|--|--------------------|---------|----------|-------------------------------------------------------|--------|--|--------|----------|---|
| E-Type | Opcode |  | Imm_address        |         | Imm_time |                                                       |        |  |        | Reserved |   |
|        | 1 byte |  | 4 bytes            |         | 2 bytes  |                                                       |        |  |        | 1 byte   |   |
| N-Type |        |  | Opcode Imm_address |         | Imm_NGID |                                                       |        |  |        | Reserved |   |
|        | 1 byte |  |                    | 4 bytes |          | 2 bytes                                               |        |  |        | 1 byte   |   |
| C-Type |        |  |                    |         |          | Opcode Imm_address Imm_CGID_col Imm_CGID_row Reserved |        |  |        |          |   |
|        | 1 byte |  | 4 bytes            |         | 1 byte   |                                                       | 1 byte |  | 1 byte |          |   |

| Category<br>Name       |   |                     | DS-ISA                        |
|------------------------|---|---------------------|-------------------------------|
| Load Node Group        | N | N_LOAD              | Data Addr, NGID               |
| Lock Node Group        |   | N_LOCK              | NLM Addr, NGID                |
| Evolve Node Groups     |   | N_EVOLVE            | GM Addr, Time                 |
| Store Node Group       | N | N_STORE             | Data Addr, NGID               |
| Load Coupling Group    | C | C_LOAD              | Data Addr, CGID_col, CGID_row |
| Lock Coupling Group    | C | C_LOCK              | CLM Addr, CGID_col, CGID_row  |
| Evolve Coupling Groups |   | C_EVOLVE            | GM Addr, Time                 |
| Store Coupling Group   | C | C_STORE             | Data Addr, CGID_col, CGID_row |
| Configure Connection   | C | CFG_CONN            | CM Addr, NGID                 |
|                        |   | Type<br>N<br>E<br>E |                               |

Fig. 6. Instruction and format definitions of DS-ISA. NLM: Node Lock Mask; GM: Group Mask; CLM: Coupling Lock Mask; CM: Connection Mask; GID: Group ID; CGID: Coupling Group ID.

This ISA follows a unique "label-and-trigger" computing mechanism, which is essential for managing synchronous, collective evolution. The N LOCK and C LOCK instructions are labeling commands that set lock masks, while the EVOLVE instructions then act as the trigger, initiating a single, collective execution by applying the pre-set masks simultaneously. This separation ensures non-locked components to begin their parallel evolution from a synchronized state.

With the set of operations defined, we must design a format that addresses the core challenge of controlling a potentially vast number of nodes and couplings. We draw inspiration from GPUs and organize nodes and couplings into groups, such that a single instruction operates on all elements within a group synchronously in lockstep. This grouping strategy simplifies the primary control problem, and a two-level, hierarchical control scheme is adopted accordingly: First, we require a mechanism for inter-group control to select which groups participate in a collective action, such as the Group Mask (GM) used to manage parallel evolution. Second, we require intragroup control for fine-grained manipulation. This includes onedimensional Node Lock Masks (NLM) for setting boundary conditions, as well as two-dimensional coupling masks. These coupling masks, such as the Coupling Lock Mask (CLM) and Connection Mask (CM), are defined by their column and row mask components to provide fine-grained control over the selected coupling group, which is specified using column and row components in Coupling Group ID (CGID).

However, this mask-based control scheme has a direct and critical implication for the instruction format: scalability. The size of these intra-group masks (e.g., NLM) scales linearly with the size of the group, and the inter-group masks (e.g., GM) scale with the number of groups. It is therefore architecturally infeasible to embed this large, variable-sized data directly into an instruction.

To resolve this, we adopt an indirect control scheme. Rather than embedding the large, scalable masks into the instruction itself, the instruction carries an address to the data in memory (e.g., on-chip SRAM). Based on this consideration, we adopt a fixed-length 64-bit instruction format, providing ample space to hold both a large address pointer and other immediate control values. This format, detailed in Figure 6, is partitioned into three distinct types (E-Type, N-Type, and C-Type) based on its operands. The 4-byte Imm address field provides a 32 bit address, which points to data and scalable masks (NLM, CLM, CM, GM) in memory. The 2-byte immediate field, in contrast, is used for data that is small or scales logarithmically, such as the Imm NGID, the Imm CGID col/Imm CGID row components, or the evolution duration Imm time. This twolevel immediate system allows our 64-bit instruction to control DSUs of extensible scale by loading the appropriate masks from memory, providing a simple, scalable, and efficient ISA.

To make a concrete connection between application and DS-ISA, Figure 7 illustrates how the DS-ISA executes a simple ML inference task following the load-lock-evolve-store model. To achieve this, input features must be mapped to input node groups, output features to output node groups, with couplings encoding the model weights that drive the influence from input to output. The procedures are: (1) To determine which nodes serve as input and output nodes, a CFG CONN instruction uses an Intra-Group Mask to CONNECT the corresponding couplings representing the influence from input to output. In this example, the highlighted coupling groups suggest that the first four node groups influence the next six node groups. (2) N LOAD and C LOAD instructions use the Memory Interface to load input data and weights into these components. (3) as inputs and weights are locked while outputs are free to evolve, N LOCK and C LOCK apply their Intra-Group Masks to lock specific members in the input node groups and corresponding coupling groups. (4) N EVOLVE instruction then uses an Inter-Group Mask to Evolve only the output node group. (5) to write back, N STORE uses the Memory Interface to Store the resulting states from the evolved nodes. This example shows how the ISA's core mechanisms, such as intra-group and intergroup masking, provide the necessary control to execute the distinct phases of our model for a practical application.

