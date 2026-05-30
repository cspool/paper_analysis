# <span id="page-2-1"></span>**3** The Omni-Expert (OE)

Our goal is to enable a single expert model to exhibit subtask-specific expertise based on the input features of the associated subtask. In sparse MoE models, a routing mechanism directs inputs to

the appropriate expert or set of experts. In contrast, our approach is to encode subtask selection for specialization *implicitly* in the feature space. To achieve this, we apply learned subtask-specific transformations that create homogenous features within a specific subtask and distinct features across subtasks. The OE model architecture, illustrated in [Figure 1b](#page-2-0), consists of three core components:

- A *feature transformation block* to apply subtask specific feature transformations based on the subtask label.
- A *single expert network*, which processes the transformed input features and performs the target task (in this case, mask estimation for speech dereverberation).
- A *gating/routing network*, which is used to weight the outputs produced by applying the single-expert network to the transformed features:

$$\hat{y} = \sum_{n=1:N} p_n(\mathbf{x}) E(\mathbf{z}_n) \tag{3}$$

where x is the input feature; pn(x), is the gating network probability of subtask n; E(zn) is the subtask-specific output from the single expert model E based on the transformed feature zn.

Instead of a simple linear transformation, an affine transformation offers greater flexibility by allowing a shift in the origin to better align the feature distribution for each subtask while preserving discriminability across subtasks. For a sparse affine transformation matrix, we restrict linear operations to scale transformations. The subtask-specific transformed feature z<sup>n</sup> is defined as:

$$\mathbf{z}_n = \mathbf{A}_n \mathbf{x} + \mathbf{b}_n \tag{4}$$

where x is the input feature; A<sup>n</sup> and b<sup>n</sup> represent scale and shift transformations, respectively, for subtask n. For scale, A<sup>n</sup> is a diagonal matrix, which simplifies to element-wise multiplication.

