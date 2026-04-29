# Appendix A — Embedded Systems Quick Reference

> Constraint quantification templates, datasheet reading guide for AI suitability, memory and power calculation worksheets.

---

## A.1 Constraint Quantification Template

Use this template when evaluating a new embedded target for AI deployment.

| Constraint | Specification | Notes |
|------------|--------------|-------|
| Flash (program storage) | ___ KB / MB | Model weights stored here |
| SRAM (runtime memory) | ___ KB / MB | Activations + tensors at inference |
| External RAM | ___ MB (if any) | |
| CPU clock | ___ MHz | |
| SIMD / DSP extensions | Yes / No | ARM CMSIS-NN, Xtensa HiFi, etc. |
| Hardware accelerator | None / NPU / DSP | |
| Supply voltage | ___ V | |
| Active current (max) | ___ mA | |
| Sleep current | ___ µA | |
| Battery capacity (if applicable) | ___ mAh | |
| Hard real-time deadline | ___ ms | |

---

## A.2 Datasheet Reading Guide for AI Suitability

<!-- TODO: step-by-step guide for reading a microcontroller datasheet -->

---

## A.3 Memory Calculation Worksheet

**Model weight memory (Flash):**
```
parameters × bytes_per_weight = flash_required
e.g., 250,000 params × 4 bytes (float32) = 1 MB
      250,000 params × 1 byte  (int8)    = 250 KB
```

**Activation memory (SRAM):**
```
peak_activation_tensor_elements × bytes_per_element = sram_required
(depends on model architecture — use profiling tool to measure)
```

---

## A.4 Battery Life Estimation Worksheet

```
inference_energy_µJ = inference_power_mW × inference_time_ms

daily_inference_energy_mJ = inference_energy_µJ × inferences_per_day / 1000

battery_life_days = (battery_capacity_mAh × supply_voltage_V × 3600) / (daily_energy_mJ / 1000)
```

---

*[Back to Table of Contents](../README.md)*
