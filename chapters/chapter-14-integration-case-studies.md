# Chapter 14 — Integration Case Studies: Full Design Decisions

> *The student applies the complete decision framework — constraint evaluation, inference analysis, model selection, and optimization — to three full embedded AI design problems.*

**Part III — Model-Aware Design**

---

## Learning Outcomes

- **LO14.1** [Evaluate] Apply the complete AI integration evaluation framework to a novel embedded design problem
- **LO14.2** [Create] Produce a structured integration decision document that specifies the model, hardware, deployment strategy, and constraint justification for a given application
- **LO14.3** [Evaluate] Critique a proposed AI embedded system design and identify the decisions most likely to fail in production

---

## Case Study A — Industrial Anomaly Detection

> **Scenario:** Vibration sensor system on manufacturing equipment.
> **Constraints:** Hard latency (10ms), battery-powered (18-month life), no cloud connectivity.

### A.1 Problem Definition
<!-- TODO -->

### A.2 Sensor Selection and Data Pipeline
<!-- TODO -->

### A.3 Model Selection
<!-- TODO -->

### A.4 Quantization and Memory Fit
<!-- TODO -->

### A.5 Hardware Selection
<!-- TODO -->

### A.6 Inference Scheduling Strategy
<!-- TODO -->

### A.7 Decisions Made, Alternatives Rejected, and When They Would Have Been Right
<!-- TODO -->

---

## Case Study B — Medical Wearable: Cardiac Arrhythmia Detection

> **Scenario:** Wearable ECG monitor with AI-based arrhythmia classification.
> **Constraints:** Safety-critical (misclassification has clinical consequences), battery-constrained, regulatory requirements (FDA SaMD pathway).

### B.1 Problem Definition
<!-- TODO -->

### B.2 Safety and Regulatory Requirements
<!-- TODO -->

### B.3 Model Selection Under Safety Constraints
<!-- TODO -->

### B.4 Reliability and Confidence Gating
<!-- TODO -->

### B.5 Power Budget
<!-- TODO -->

### B.6 Decisions Made, Alternatives Rejected, and When They Would Have Been Right
<!-- TODO -->

---

## Case Study C — Agricultural Edge Sensor: Crop Disease Detection

> **Scenario:** Low-cost, solar-powered image classification system for disease detection in the field.
> **Constraints:** Intermittent connectivity, energy harvesting, low unit cost.

### C.1 Problem Definition
<!-- TODO -->

### C.2 Edge-Cloud Decision
<!-- TODO -->

### C.3 Hardware Selection Under Cost Constraint
<!-- TODO -->

### C.4 Model Selection and Optimization
<!-- TODO -->

### C.5 Energy Harvesting Integration
<!-- TODO -->

### C.6 Decisions Made, Alternatives Rejected, and When They Would Have Been Right
<!-- TODO -->

---

## Final Note

> Each case study ends with an explicit list of decisions that were made, the alternatives that were rejected, and the conditions under which a rejected alternative would have been the right choice. This is the evaluation skill the book was built to develop.

---

*Previous: [Chapter 13](./chapter-13-tinyml-toolchains.md) | [Appendix A](./appendix-a-embedded-quick-reference.md)*
