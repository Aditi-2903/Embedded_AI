# Appendix D — Toolchain Setup Guides

> Step-by-step setup for TensorFlow Lite for Microcontrollers, Edge Impulse, and STM32Cube.AI.

---

## D.1 TensorFlow Lite for Microcontrollers

### Requirements
- Python 3.8+
- TensorFlow 2.x

### Installation

```bash
pip install tensorflow
```

### Convert a Model to TFLite

```python
import tensorflow as tf

# Load your saved model
converter = tf.lite.TFLiteConverter.from_saved_model('my_model/')

# Optional: apply post-training quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Convert to C Array for Microcontroller

```bash
xxd -i model.tflite > model_data.cc
```

<!-- TODO: expand with Arduino deployment steps -->

---

## D.2 Edge Impulse

<!-- TODO: account setup, data collection, model training, export to Arduino/STM32 -->

---

## D.3 STM32Cube.AI

<!-- TODO: STM32CubeIDE installation, X-CUBE-AI pack, model import, validation steps -->

---

*[Back to Table of Contents](../README.md)*
