# Zero Framework API Shell

> **Note:** This repository is an API-compatible shell. All underlying math, autodiff, and graph execution has been migrated to the [ml-switcheroo-compiler](https://github.com/SamuelMarks/ml-switcheroo-compiler) backend. This repository purely implements frontend routing and syntactic parity for the target framework.

# zero-orbax (v0.6.4)

[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/SamuelMarks/zero-orbax/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/zero-orbax/actions)
[![Test Coverage](https://img.shields.io/badge/test_coverage-100%25-brightgreen.svg)](#)
[![Doc Coverage](https://img.shields.io/badge/doc_coverage-100%25-brightgreen.svg)](#)
[![API Compliance](https://img.shields.io/badge/api_compliance-100%25-brightgreen.svg)](#)

**zero-orbax** is a fully implemented, structurally identical, and zero-dependency drop-in replacement for the [`orbax`](https://github.com/google/orbax) library, specifically targeting the [`orbax.checkpoint`](https://orbax.readthedocs.io/en/latest/api_reference/checkpoint.html) API. The semantic logic, including saving, restoring, async futures, and PyTree serialization, is completely functional and verified against the official `orbax` test suite. This project is meticulously synchronized with [`orbax`](https://github.com/google/orbax) version **0.6.4**.

---

## Why does this project exist?

This repository is a core Tier 4 component of the [`ml-switcheroo`](https://github.com/SamuelMarks/ml-switcheroo) ecosystem. The [`ml-switcheroo`](https://github.com/SamuelMarks/ml-switcheroo) project solves the $N \times M$ translation problem in [Machine Learning](https://en.wikipedia.org/wiki/Machine_learning) by tracing $N$ framework frontends ([JAX](https://github.com/google/jax), [PyTorch](https://pytorch.org/), [Keras](https://keras.io/)) into a unified [Intermediate Representation (IR)](https://en.wikipedia.org/wiki/Intermediate_representation), and compiling that IR into $M$ backends ([WASM](https://webassembly.org/), [WebGPU](https://www.w3.org/TR/webgpu/)).

Standard ML frameworks carry massive dependency trees. Importing the real [`orbax`](https://github.com/google/orbax) pulls in [JAX](https://github.com/google/jax), [Flax](https://github.com/google/flax), [TensorStore](https://google.github.io/tensorstore/), and gigabytes of associated native binaries. This completely breaks source-to-source tracing, lightweight CI validations, and [WebAssembly](https://webassembly.org/)/browser compilation pipelines where environments are highly constrained.

**`zero-orbax` solves this by providing:**
1. **100% Structural API Compliance:** Checked automatically against [`ml-framework-snapshots`](https://github.com/SamuelMarks/ml-framework-snapshots). When a user types `from orbax.checkpoint import CheckpointManager`, the system seamlessly routes this to `zero_orbax.checkpoint.CheckpointManager` without raising any signature, type, or attribute errors.
2. **Zero External Dependencies:** Built entirely on the [Python Standard Library](https://docs.python.org/3/library/) (and [`numpy`](https://numpy.org/)). No heavy binaries, no C-extensions.
3. **Seamless State Tracing:** Checkpoint reading and writing operations are intercepted. Instead of strictly persisting physical [`msgpack`](https://msgpack.org/) files during an AOT compile, `zero-orbax` natively supports [PyTree](https://jax.readthedocs.io/en/latest/pytrees.html) processing, deterministic dictionary merging (`merge_trees`), and mock resolution via synchronous [`Future`](https://docs.python.org/3/library/concurrent.futures.html) classes to serialize the [`PyTree`](https://jax.readthedocs.io/en/latest/pytrees.html) structure safely into the compiler's logical graph.

### The Role of State Persistence
In the [`ml-switcheroo`](https://github.com/SamuelMarks/ml-switcheroo) ecosystem, state variables and parameters need to be "lifted" to functional purity. `zero-orbax` plays the critical role of allowing existing open-source training scripts (which heavily use [`orbax.checkpoint`](https://orbax.readthedocs.io/en/latest/api_reference/checkpoint.html) to save their model parameters) to run unmodified while capturing these states for browser export.

---

## License

Licensed under either of

- [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- [MIT license](https://opensource.org/licenses/MIT) ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0), shall be
dual licensed as above, without any additional terms or conditions.