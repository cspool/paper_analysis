# II. BACKGROUND

In this section, we describe the concepts and terminology used in the remainder of the paper.

## A. Tensors

This paper focuses on algebraic computations on tensors, where a tensor is a multidimensional array. A tensor's rank refers to a specific dimension of the tensor, while the tensor's shape is the set of valid coordinates for each of the tensor's ranks. We use the notation N-tensor to denote a tensor with N ranks, where a 0-tensor is a scalar, a 1-tensor is a vector, a 2-tensor is a matrix, etc.

We adopt the format-agnostic *fibertree* abstraction of tensors, where a tensor is represented as a tree of fibers, as detailed in prior work [25], [35], [38], [43], [51], [55], [57], [58], using the specific version described in TeAAL [35, Section 2.1]. In this abstraction, a *fiber* consists of the set of coordinates for a given rank with common coordinates for all higher-level ranks. Each coordinate is coupled with a *payload*. The payload may contain a reference to a fiber in the next lower rank, or to a leaf data *value*.

