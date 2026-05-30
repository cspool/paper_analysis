# <span id="page-2-1"></span>3 Fix: describing dataflow to the runtime

In this section, we describe the design of Fix: a low-level binary representation, or ABI, where code externalizes its dataand control flow. Instead of fetching data by making network connections or syscalls and waiting for a reply, programs describe the code and data they need declaratively, in a format that's parsed and executed by the runtime infrastructure.

Fix objects represent pieces of data, function invocations, dependencies, and data sub-selection, in an in-memory representation that is independent of programming language and placement on a server. User functions make dataflow visible by constructing Fix objects; a runtime can exchange references to these objects with native functions using a defined calling convention. The computation graph necessary to evaluate a Fix object is described by the object itself, so runtimes do not need to maintain additional metadata.

Fix's design is intended to let pieces of black-box machine code precisely express their data needs in a manner lightweight enough to permit microsecond-level overheads, but general enough to support arbitrary applications, including ones where the dataflow graph evolves over the course of a computation in a data-dependent way. To enable efficient and flexible execution, Fix's design goals were:

- 1. Code can be represented as black-box machine code that originated from any programming language.
- 2. The complete data "footprint" needed to evaluate a function call will be known before it is invoked.
- 3. A function will always run to completion without blocking, and will finish execution without invoking another function or enlarging its data "footprint."
- 4. Functions will have tools to subselect from large data structures to fetch only the portion truly needed.

These considerations led to the design below.

