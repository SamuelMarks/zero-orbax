# zero-orbax (v0.6.4)

[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/SamuelMarks/zero-orbax/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/zero-orbax/actions)
[![Test Coverage](https://img.shields.io/badge/test_coverage-100%25-brightgreen.svg)](#)
[![Doc Coverage](https://img.shields.io/badge/doc_coverage-100%25-brightgreen.svg)](#)

**zero-orbax** is a structurally identical, zero-dependency drop-in replacement for the `orbax` library, specifically targeting the `orbax.checkpoint` API. This project is meticulously synchronized with `orbax` version **0.6.4**.

---

## Why does this project exist?

This repository is a core Tier 4 component of the `ml-switcheroo` ecosystem. The `ml-switcheroo` project solves the $N \times M$ translation problem in Machine Learning by tracing $N$ framework frontends (JAX, PyTorch, Keras) into a unified Intermediate Representation (IR), and compiling that IR into $M$ backends (WASM, WebGPU).

Standard ML frameworks carry massive dependency trees. Importing the real `orbax` pulls in JAX, Flax, TensorStore, and gigabytes of associated native binaries. This completely breaks source-to-source tracing, lightweight CI validations, and WebAssembly/browser compilation pipelines where environments are highly constrained.

**`zero-orbax` solves this by providing:**
1. **100% Structural API Compliance:** Checked automatically against `ml-framework-snapshots`. When a user types `from orbax.checkpoint import CheckpointManager`, the system seamlessly routes this to `zero_orbax.checkpoint.CheckpointManager` without raising any signature, type, or attribute errors.
2. **Zero External Dependencies:** Built entirely on the Python Standard Library (and `numpy`). No heavy binaries, no C-extensions.
3. **Seamless State Tracing:** Checkpoint reading and writing operations are intercepted. Instead of strictly persisting physical `msgpack` files during an AOT compile, `zero-orbax` natively supports PyTree processing, deterministic dictionary merging (`merge_trees`), and mock resolution via synchronous `Future` classes to serialize the `PyTree` structure safely into the compiler's logical graph.

### The Role of State Persistence
In the `ml-switcheroo` ecosystem, state variables and parameters need to be "lifted" to functional purity. `zero-orbax` plays the critical role of allowing existing open-source training scripts (which heavily use `orbax.checkpoint` to save their model parameters) to run unmodified while capturing these states for browser export.

---

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.