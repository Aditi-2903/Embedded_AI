# Chapter 5 — Memory: Fitting Models into Constrained Spaces

> *The student learns to analyze and reduce the memory footprint of AI models for embedded deployment.*

**Part II — Inference-First Evaluation**

---

## Learning Outcomes

- **LO5.1** [Apply] Calculate the flash and SRAM requirements of a given model for a specified target
- **LO5.2** [Analyze] Identify which components of a model consume the most memory and why
- **LO5.3** [Evaluate] Determine whether a model fits a target's memory envelope and specify what must change if it does not

---

## 5.1 Flash vs. SRAM in AI Deployment

> *Where weights live, where activations live, and why both matter*

<!-- TODO -->

---

## 5.2 Weight Memory

> *Parameter count × precision = storage requirement*

<!-- TODO -->

---

## 5.3 Activation Memory

> *The hidden footprint — how intermediate tensors consume SRAM during inference*

<!-- TODO -->

---

## 5.4 Memory Layout Strategies

> *In-place computation, buffer reuse, scratch pads*

<!-- TODO -->

---

## 5.5 Static Memory Allocation for Inference

> *Why malloc is not your friend*

<!-- TODO -->

---

## 5.6 Memory Profiling Tools

> *How to measure what a deployed model actually uses*

<!-- TODO -->

---

## Worked Example: Activation Memory Overflow

> *A model that "fits" by parameter count fails at runtime due to activation memory. Diagnose, then apply buffer reuse to fix it.*

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

*Previous: [Chapter 4](./chapter-04-inference-mechanics.md) | Next: [Chapter 6 — Compute](./chapter-06-compute.md)*
