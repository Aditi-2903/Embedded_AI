# Chapter 10 — Real-Time AI: Reliability, Safety, and Deadlines

> *The student learns to evaluate whether AI inference can be integrated into real-time embedded systems without violating timing guarantees or safety requirements.*

**Part II — Inference-First Evaluation**

---

## Learning Outcomes

- **LO10.1** [Analyze] Identify the properties of AI inference that make WCET analysis difficult and explain why determinism matters for real-time systems
- **LO10.2** [Evaluate] Assess whether a proposed AI component can be safely integrated into a hard real-time system
- **LO10.3** [Apply] Apply design patterns that reduce the real-time risk of AI inference in safety-critical applications

---

## Opening: A Hard Case

> *An autonomous vehicle that fails to meet a braking deadline because the inference system ran long. Why did WCET analysis miss it? What should the designer have done differently?*

<!-- TODO -->

---

## 10.1 Hard vs. Soft Real-Time Requirements

> *What they demand from AI inference*

<!-- TODO -->

---

## 10.2 Non-Determinism in Neural Network Inference

> *Why the same model can have variable execution time*

<!-- TODO -->

---

## 10.3 WCET Analysis for AI

> *What works, what doesn't, and what the open problems are*

> ⚠️ **Open Question OQ-002:** This is an active research area with no settled practice. This chapter stakes a position that needs expert review before publication.

<!-- TODO -->

---

## 10.4 Safety-Critical AI

> *Functional safety standards (IEC 61508, ISO 26262) and what they require of AI components*

<!-- TODO -->

---

## 10.5 Design Patterns for Real-Time-Safe AI

> *Bounded execution models, rejection thresholds, confidence gating*

<!-- TODO -->

---

## 10.6 Failure Mode Analysis for AI in Embedded Systems

<!-- TODO -->

---

## Worked Example: Neural Network Anomaly Detector in a Hard Real-Time Control Loop

> *Identify the failure modes, apply a confidence gating pattern, verify the timing guarantee.*

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

*Previous: [Chapter 9](./chapter-09-communication-edge-cloud.md) | Next: [Chapter 11 — Selecting Models for Constrained Deployment](./chapter-11-model-selection.md)*
