# river-cpp

> **A High-Performance, Header-only C++20 Streaming Learning Library.**
> *Porting the logic of [River](https://github.com/online-ml/river) to C++.*

![Standard](https://img.shields.io/badge/Standard-C%2B%2B20-blue.svg?style=flat-square&logo=c%2B%2B)
![Build](https://img.shields.io/badge/Build-Header--only-green.svg?style=flat-square)
![License](https://img.shields.io/badge/License-BSD_3--Clause-yellow.svg?style=flat-square)

**river-cpp** is a modern C++ implementation of online machine learning algorithms, engineered for high-performance computing, embedded systems, and real-time infrastructure where the Python interpreter overhead is unacceptable.

## Why river-cpp?

While the original Python `river` library is excellent for prototyping and algorithmic research, integrating it into low-latency C++ storage or infrastructure systems introduces significant costs: **GIL contention, data serialization, and context switching**.

**river-cpp solves this by providing:**

* **Zero-Overhead:** Pure C++ implementation with **no Python dependency**. Directly linkable into storage engines or kernels.
* **Header-Only:** Seamless integration into existing build systems (CMake/Bazel/Make) by simply including headers.
* **Modern C++ Design:** Built with **C++20 Concepts** and **Templates** to achieve static polymorphism (avoiding virtual function costs in hot paths).

## Architecture & Concepts

`river-cpp` is designed as a **framework**, not just a collection of algorithms. It decouples the learning logic using strict compile-time contracts.

### The `rivercpp::IsDetector` Concept (The Hook)
Any algorithm that satisfies the `rivercpp::IsDetector` concept can be plugged into the evaluation pipeline. This ensures type safety and enables compiler optimizations (inlining).

```cpp
namespace rivercpp {
template <typename D>
concept IsDetector = requires(D detector, double value) {
    { detector.update(value) } -> std::same_as<void>;
    { detector.drift_detected } -> std::convertible_to<bool>;
    detector.drift_detected = true;

    requires std::movable<D>;
};
}

```

## Performance Benchmarks（TODO）

*Hardware: Intel Core i9 / Dataset: Synthetic Stream (Agrawal Generator, 1M samples)*

| Implementation | Time (1M samples) | Memory Usage | Bottleneck |
| --- | --- | --- | --- |
| **River (Python)** | ? | High (Dynamic typing) | Interpreter Loop + GIL |
| **river-cpp (ARF)** | **?** | **Low (Compact struct)** | **None (CPU Bound)** |

> **Result:** `river-cpp` achieves approximately **?x throughput improvement** compared to the Python reference implementation in pure sequential training loops.

## Implemented Algorithms

* **Hoeffding Tree**
  * Full C++ rewrite of `river.tree.HoeffdingTreeClassifier`.

* **Adaptive Random Forest (ARF)**
  * Full C++ rewrite of `river.forest.ARFClassifier`.
  * Features **Dynamic Drift Detection** (ADWIN/DDM) and background tree training.

* *Streaming Naive Bayes (Planned)*

## Quick Start

No build tools required. Just include the header.

**Note on Naming:** The project is named `river-cpp` to pay tribute to the Python library, but the internal namespace is `rivercpp` for cleaner C++ code.

```cpp
// to be done.

```

## Attribution & License

This project is a derivative work of **[River](https://github.com/online-ml/river)**.

* **Algorithm Logic:** Derived from River (Copyright by River Authors).
* **System Architecture:** Re-designed by Apolynoid for C++ systems optimization.

Licensed under the **BSD-3-Clause License** (Same as River). Original algorithm copyright (c) belongs to the River project authors.

```
