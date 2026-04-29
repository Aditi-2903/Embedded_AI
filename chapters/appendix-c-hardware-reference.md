# Appendix C — Hardware Reference: AI-Capable Microcontroller and SoC Comparison

> ⚠️ **Open Question OQ-004:** This appendix has the highest risk of becoming outdated. Recommend separating into online-only companion content updated annually.

---

## C.1 Comparison Table

| Device | Core | Flash | SRAM | Accelerator | Peak Freq | Est. Power | Toolchain |
|--------|------|-------|------|-------------|-----------|------------|-----------|
| Arduino Nano 33 BLE Sense | Cortex-M4 | 1 MB | 256 KB | None | 64 MHz | ~50 mW | TFLite Micro |
| STM32F411 | Cortex-M4F | 512 KB | 128 KB | None | 100 MHz | ~100 mW | STM32Cube.AI |
| STM32H7B3 | Cortex-M7 | 2 MB | 1.4 MB | None | 280 MHz | ~300 mW | STM32Cube.AI |
| STM32N6 | Cortex-M55 | 4 MB | 4 MB | Ethos-U55 | 800 MHz | ~200 mW | STM32Cube.AI |
| ESP32-S3 | Xtensa LX7 × 2 | 16 MB (ext.) | 512 KB | AI ext. | 240 MHz | ~240 mW | ESP-DL |
| Kendryte K210 | RISC-V × 2 | External | 8 MB | KPU | 400 MHz | ~300 mW | NNCase |
| Coral Edge TPU (USB) | — (coprocessor) | — | — | Google TPU | — | ~2 W | TFLite |
| Raspberry Pi Zero 2W | Cortex-A53 × 4 | microSD | 512 MB | None | 1 GHz | ~700 mW | TFLite / ONNX |
| Raspberry Pi 4 | Cortex-A72 × 4 | microSD | 1–8 GB | None | 1.8 GHz | ~3–7 W | Full Python stack |

---

## C.2 Selection Guide

- **Tightest constraints (< 256 KB SRAM):** STM32F4, Arduino Nano 33 → quantized models only, TFLite Micro
- **Mid-range (256 KB – 2 MB SRAM):** STM32H7, ESP32-S3 → more model options, mixed precision
- **Hardware accelerator needed:** STM32N6 (Ethos-U55), Kendryte K210 (KPU), Coral Edge TPU
- **Linux-capable edge:** Raspberry Pi family → full toolchain available

---

*[Back to Table of Contents](../README.md)*
