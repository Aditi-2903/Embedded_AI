# Chapter 3 — Machine Learning for Embedded Engineers: A Functional Foundation

> *The student builds enough ML literacy to reason about inference requirements without needing to train models.*

**Part I — The Embedded-AI Problem**

---

## Learning Outcomes

- **LO3.1** [Understand] Explain supervised learning, inference, and the trained model as artifacts with specific resource profiles
- **LO3.2** [Apply] Identify the computational operations a neural network performs during inference
- **LO3.3** [Analyze] Map neural network architecture choices to memory and compute requirements
- **LO3.4** [Evaluate] Assess which class of ML model (CNN, RNN, decision tree, etc.) is appropriate for a given embedded sensing task

---

> **Critical design note:** This chapter teaches ML from the perspective of what inference *costs*, not how training works. A student who finishes this chapter cannot train a model — but can reason about what any trained model will demand from constrained hardware.

---

## Opening: What Is Inside a `.tflite` File?

> *Open one. The student sees weights, graph structure, input/output tensors. This is the artifact they will deploy.*

<!-- TODO -->

---

## 3.1 Supervised Learning: The Model as a Fixed Artifact

> *Inputs, labels, training, and the trained model as a fixed artifact*

<!-- TODO -->

---

## 3.2 Neural Network Anatomy

> *Layers, weights, activations — as memory and compute objects*

<!-- TODO -->

---

## 3.3 The Inference Pass

> *What happens when input data enters the model (matrix multiply, activation, repeat)*

<!-- TODO -->

---

## 3.4 Common Architectures for Embedded Sensing

> *CNNs for image, RNNs/LSTMs for sequence, MobileNet family, decision trees*

<!-- TODO -->

---

## 3.5 Model Size, Parameter Count, and FLOP Count

> *Why they matter differently*

<!-- TODO -->

---

## 3.6 The ML-to-Embedded Translation Table

> *ML terms mapped to embedded constraint terms*

| ML Term | Embedded Meaning |
|---------|-----------------|
| Parameters / Weights | Flash storage requirement |
| Activations | SRAM requirement during inference |
| FLOPs | Compute cycles / latency |
| Batch size | Not applicable at inference on embedded (always 1) |
| Precision (float32/int8) | Memory × compute multiplier |

<!-- TODO: expand -->

---

## Worked Example: MobileNetV2 on a Constrained Target

> *Walk through parameter count, memory footprint, and inference FLOP count before writing a single line of code.*

<!-- TODO -->

---

## Summary

<!-- TODO -->

---

## Review Questions

1. <!-- TODO -->
2. <!-- TODO -->
3. <!-- TODO -->

---

*Previous: [Chapter 2](./chapter-02-embedded-constraints-as-design-variables.md) | Next: [Chapter 4 — Inference Mechanics](./chapter-04-inference-mechanics.md)*
