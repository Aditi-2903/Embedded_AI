# Chapter 4 — Inference Mechanics: What a Deployed Model Actually Does

> *The student learns to trace the inference execution path on constrained hardware and identify where failures occur.*

**Part II — Inference-First Evaluation**

---

## Learning Outcomes

- **LO4.1** [Apply] Trace the inference execution path of a deployed model on a target microcontroller
- **LO4.2** [Analyze] Identify the computational bottleneck in a given model-hardware pairing
- **LO4.3** [Evaluate] Determine whether a proposed model-hardware configuration can meet a specified inference latency requirement

---

## Opening

> *Take the model from Chapter 3. Deploy it. Watch it fail — wrong latency, memory overflow, or incorrect output. The student has all the vocabulary to diagnose this now. This chapter gives them the tools.*

<!-- TODO -->

---

## 4.1 The Inference Execution Pipeline

> *On bare-metal and RTOS-based systems*

<!-- TODO -->

---

## 4.2 Operator-Level Execution

> *How convolution, matrix multiply, and activation functions run on CPU vs. specialized hardware*

<!-- TODO -->

---

## 4.3 Memory Allocation During Inference

> *Static vs. dynamic, and why dynamic allocation fails on embedded targets*

<!-- TODO -->

---

## 4.4 Latency Profiling

> *How to measure and interpret inference time on target hardware*

<!-- TODO -->

---

## 4.5 Numerical Precision

> *float32 vs. float16 vs. int8 and what the tradeoffs are*

| Format | Bits | Relative Size | Notes |
|--------|------|---------------|-------|
| float32 | 32 | 1× | Default training format |
| float16 | 16 | 0.5× | Limited embedded support |
| int8 | 8 | 0.25× | Primary embedded target |

<!-- TODO: expand -->

---

## Worked Example: Keyword Spotting on Arduino Nano 33 BLE Sense

> *Measure latency, identify the bottleneck operation, propose one mitigation.*

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

*Previous: [Chapter 3](./chapter-03-ml-for-embedded-engineers.md) | Next: [Chapter 5 — Memory](./chapter-05-memory.md)*
