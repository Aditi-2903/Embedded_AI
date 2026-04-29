# Chapter 12 — Optimizing Models: Quantization, Pruning, and Distillation

> *The student learns to apply model compression techniques and evaluate the tradeoffs of each for embedded deployment.*

**Part III — Model-Aware Design**

---

## Learning Outcomes

- **LO12.1** [Apply] Apply post-training quantization to a model and measure the accuracy-efficiency tradeoff
- **LO12.2** [Analyze] Compare quantization, pruning, and knowledge distillation on their effects on model size, latency, power, and accuracy
- **LO12.3** [Evaluate] Select and apply the appropriate compression technique(s) for a specified model-hardware-accuracy constraint set

---

## 12.1 Quantization

> *float32 → int8, what changes, what breaks, and how to measure the damage*

<!-- TODO -->

---

## 12.2 Post-Training Quantization vs. Quantization-Aware Training

> *When each is appropriate*

<!-- TODO -->

---

## 12.3 Pruning

> *Structured vs. unstructured, and why structured pruning is the embedded-relevant version*

<!-- TODO -->

---

## 12.4 Knowledge Distillation

> *Training a small model to mimic a large model (conceptual + practical)*

<!-- TODO -->

---

## 12.5 Combining Techniques

> *Quantize then prune, distill then quantize — ordering matters*

<!-- TODO -->

---

## 12.6 The Accuracy Floor

> *How to know when further compression will break the application*

<!-- TODO -->

---

## Worked Example: Memory-Failing Model — Quantize, Prune, Evaluate

> *Apply post-training quantization. Measure accuracy loss. Apply structured pruning. Re-measure. Determine whether the application requirement is still met.*

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

*Previous: [Chapter 11](./chapter-11-model-selection.md) | Next: [Chapter 13 — TinyML Toolchains](./chapter-13-tinyml-toolchains.md)*
