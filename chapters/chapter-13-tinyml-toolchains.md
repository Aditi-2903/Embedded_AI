# Chapter 13 — TinyML Toolchains and Deployment Pipelines

> *The student learns to navigate the end-to-end toolchain from trained model to deployed firmware and identify where decisions in the pipeline affect the final system.*

**Part III — Model-Aware Design**

---

## Learning Outcomes

- **LO13.1** [Apply] Convert a trained model to a deployment format (TFLite, ONNX) and deploy it to a target microcontroller using a standard toolchain
- **LO13.2** [Analyze] Identify the decisions made by the toolchain that affect inference accuracy, latency, and memory usage
- **LO13.3** [Evaluate] Select an appropriate toolchain for a specified hardware target and deployment requirement

---

> ⚠️ **Open Question OQ-003:** Toolchain coverage will age fastest. TFLite for Microcontrollers, Edge Impulse, and STM32Cube.AI are current as of 2024–2025. Consider a companion website for toolchain updates between editions.

---

## 13.1 The Deployment Pipeline

> *Train → Convert → Optimize → Compile → Deploy → Verify*

<!-- TODO -->

---

## 13.2 TensorFlow Lite for Microcontrollers

> *The dominant path and its constraints*

<!-- TODO -->

---

## 13.3 Edge Impulse

> *End-to-end toolchain for embedded ML — what it handles, what it hides*

<!-- TODO -->

---

## 13.4 ONNX and Vendor-Specific Compilers

> *When the generic path doesn't work*

<!-- TODO -->

---

## 13.5 Toolchain-Induced Accuracy Loss

> *What the converter does that the designer needs to understand*

<!-- TODO -->

---

## 13.6 Testing and Validation After Deployment

> *Why the model that passed accuracy tests may fail on device*

<!-- TODO -->

---

## Worked Example: Full Pipeline — Keyword Spotting on Arduino Nano 33 BLE Sense

> *Train in TensorFlow → Convert to TFLite → Optimize with post-training quantization → Deploy → Verify inference accuracy on device.*

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

*Previous: [Chapter 12](./chapter-12-model-optimization.md) | Next: [Chapter 14 — Integration Case Studies](./chapter-14-integration-case-studies.md)*
