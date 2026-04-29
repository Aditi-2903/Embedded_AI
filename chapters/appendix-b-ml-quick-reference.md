# Appendix B — Machine Learning Quick Reference

> Core ML concepts, common architectures, vocabulary table. For students who want a reference without re-reading Chapter 3.

---

## B.1 Core Vocabulary

| Term | Definition | Embedded Relevance |
|------|------------|-------------------|
| Parameter / Weight | A learned numerical value in the model | Stored in Flash |
| Activation | Intermediate output of a layer during inference | Stored in SRAM |
| Layer | A computational unit in a neural network | Determines compute structure |
| Inference | Running a trained model on new input | The only operation on embedded |
| Training | Optimizing weights using labeled data | Done off-device |
| FLOP | Floating-point operation | Proxy for compute cost |
| MAC | Multiply-accumulate operation | 1 MAC ≈ 2 FLOPs |
| Quantization | Reducing numerical precision (float32 → int8) | Reduces memory and compute |
| Pruning | Removing low-importance weights | Reduces model size |
| Distillation | Training small model to mimic large model | Produces efficient models |

---

## B.2 Common Architectures for Embedded

| Architecture | Task | Parameter Range | Notes |
|-------------|------|-----------------|-------|
| MobileNetV2 | Image classification | 3.4M | Designed for mobile/edge |
| MobileNetV3 | Image classification | 5.4M | More efficient than V2 |
| EfficientNet-Lite | Image classification | 4–13M | Embedded-optimized variant |
| SqueezeNet | Image classification | 1.2M | Very small, older design |
| YOLO-nano | Object detection | ~1M | Minimal detection model |
| DS-CNN | Keyword spotting | ~500K | MLCommons reference model |
| TCN | Time-series | Varies | Temporal Convolutional Network |
| Decision Tree | Classification | Varies | Deterministic, very fast |

---

## B.3 Precision Format Reference

| Format | Bits | Bytes per weight | Typical accuracy drop |
|--------|------|------------------|-----------------------|
| float32 | 32 | 4 | Baseline |
| float16 | 16 | 2 | ~0% |
| int8 | 8 | 1 | 0–2% with PTQ |
| int4 | 4 | 0.5 | 1–5% |

---

*[Back to Table of Contents](../README.md)*
