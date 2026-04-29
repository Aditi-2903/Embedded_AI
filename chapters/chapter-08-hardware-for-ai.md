# Chapter 8 — Hardware for AI: Accelerators, DSPs, and FPGAs

> *The student learns to evaluate hardware acceleration options and select the appropriate acceleration strategy for a constrained AI deployment.*

**Part II — Inference-First Evaluation**

---

## Learning Outcomes

- **LO8.1** [Understand] Explain how neural processing units (NPUs), DSPs, and FPGAs accelerate AI inference differently
- **LO8.2** [Analyze] Compare the memory, compute, power, and flexibility tradeoffs of different acceleration architectures
- **LO8.3** [Evaluate] Select an appropriate hardware acceleration strategy for a given application's constraint profile

---

## 8.1 Why CPUs Are Inefficient for Neural Network Inference

> *The mismatch explained*

<!-- TODO -->

---

## 8.2 DSPs for AI

> *Fixed-point arithmetic, SIMD, and signal processing pipelines*

<!-- TODO -->

---

## 8.3 NPUs and MicroNPUs

> *What they are, what they accelerate well, and what they don't*

<!-- TODO -->

---

## 8.4 FPGAs for Embedded AI

> *Reconfigurability vs. power vs. development cost*

<!-- TODO -->

---

## 8.5 Integrated AI Hardware: Real Devices, Real Tradeoffs

> *STM32N6, Kendryte K210, Coral Edge TPU*

| Device | Accelerator | Peak TOPS | Power | Notes |
|--------|------------|-----------|-------|-------|
| STM32N6 | Ethos-U55 NPU | ~0.5 | ~100mW | ARM ecosystem |
| Kendryte K210 | KPU | ~0.23 | ~300mW | RISC-V |
| Coral Edge TPU | Google TPU | 4 | ~2W | USB/PCIe |

<!-- TODO: expand -->

---

## 8.6 When NOT to Use a Hardware Accelerator

> *The cases where a well-optimized CPU model wins*

<!-- TODO -->

---

## Worked Example: Same Model, Three Hardware Targets

> *Cortex-M4 CPU, Cortex-M55 + Ethos-U55 NPU, Xilinx PYNQ FPGA — compare latency, power, and development effort.*

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

*Previous: [Chapter 7](./chapter-07-power-and-energy.md) | Next: [Chapter 9 — Communication and the Edge-Cloud Spectrum](./chapter-09-communication-edge-cloud.md)*
