# Chapter 13: TinyML Toolchains and Deployment Pipelines

You have a trained model. You have quantized it to int8. You have pruned 25% of its channels. The model achieves 90% accuracy on your validation set. Now you need to convert it from a training artifact (a .h5 file in Keras or a .pth file in PyTorch) into firmware that runs on your STM32H7 microcontroller.

You attempt the conversion using TensorFlow Lite's converter. The converter runs, produces a .tflite file, and reports success. You integrate the .tflite file into your firmware using TensorFlow Lite for Microcontrollers, compile, flash, and run.

The first inference produces garbage output—predictions that don't match the validation results. You debug for hours and discover the issue: the converter applied dynamic range quantization to layers you thought were already int8, and the resulting double-quantization destroyed accuracy. The model that worked perfectly in Python fails silently on the target hardware.

This is the deployment toolchain problem. The path from trained model to running firmware involves multiple tools—converters, optimizers, compilers, and runtime libraries—and each tool makes decisions that affect accuracy, latency, and memory usage. Some of those decisions are visible, some are hidden, and some are wrong. Understanding the toolchain is the prerequisite for successful deployment.

This chapter teaches you to navigate the end-to-end deployment pipeline from trained model to verified on-target inference, identify the decisions made by converters and compilers that affect deployment, select the appropriate toolchain for your hardware target and model format, and test and validate that the deployed model produces correct results. By the end of this chapter, you will be able to convert a model, deploy it, and verify that it works before it reaches production.

The Deployment Pipeline: Train → Convert → Optimize → Compile → Deploy → Verify

Deployment is not a single step. It's a sequence of transformations:

Step 1: Train

 You train a model in TensorFlow, PyTorch, or another framework. The output is a training checkpoint or saved model (.h5, .pth, .onnx, .pb). This model runs on a GPU or CPU with Python, NumPy, and the full framework runtime.

## Step 2: Convert

 You convert the training model to a deployment format. The converter removes training-specific operations (dropout, batch normalization folding), applies graph optimizations (operator fusion, constant propagation), and prepares the model for the target runtime. The output is a deployment model (.tflite, .onnx, vendor-specific formats).

Step 3: Optimize (optional but common)

 The optimizer applies quantization, pruning, or layer fusion to reduce model size or latency. Some converters include optimization (TFLite converter does quantization), others require separate tools (Vela for ARM Ethos-U, TensorRT for NVIDIA).

## Step 4: Compile

 The deployment model is compiled into firmware. For microcontrollers, this means linking the model file with an inference engine (TFLite Micro, CMSIS-NN) and your application code, then compiling to a binary with GCC or ARM Compiler.

## Step 5: Deploy

 The firmware binary is flashed to the target device via JTAG, USB, or serial bootloader.

## Step 6: Verify

 You run inference on the target and compare outputs to the training model's outputs on the same inputs. If outputs match (within quantization error), deployment succeeded. If they don't, you trace back to find which step introduced the error.

This pipeline has failure modes at every step. The converter may fail to convert a custom layer. The optimizer may apply an incompatible optimization. The compiler may miscompile a kernel. The runtime may have a bug. Verification catches these failures before they reach production.

## TensorFlow Lite for Microcontrollers: The Dominant Path

TensorFlow Lite for Microcontrollers (TFLite Micro) is the most widely used inference engine for embedded AI. It's a C++ library with no dependencies on the standard library (std::vector, std::map, etc.), no dynamic memory allocation (configurable), and a small code footprint (~100 KB).

The TFLite Conversion Process

Input: A trained TensorFlow or Keras model.

Output: A .tflite file (FlatBuffers format) containing the model graph, weights, and metadata.

Converter invocation:

import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model('my_model.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable int8 quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.representative_dataset = representative_dataset_gen  # Calibration data

# Convert
tflite_model = converter.convert()

# Save
with open('my_model.tflite', 'wb') as f:

    f.write(tflite_model)

What the converter does:

Extracts the model graph (layers, connections, parameters).

Applies graph optimizations (fuses layers, removes unused nodes).

Converts weights to int8 (if quantization is enabled).

Inserts quantization/dequantization ops between layers if needed.

Serializes to FlatBuffers format.

What can go wrong:

Custom layers not supported by TFLite (e.g., custom activation functions, third-party layers).

Quantization applied incorrectly (wrong calibration data, mismatched ranges).

Converter fails silently (produces a .tflite file but with wrong layer configurations).

Deploying TFLite Micro on a Microcontroller

Step 1: Integrate the model file.

 The .tflite file is a binary blob. You include it in your firmware as a byte array:

// Generated from xxd -i my_model.tflite > my_model_data.h

alignas(16) const unsigned char my_model_data[] = {

    0x18, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, ...

};

const unsigned int my_model_data_len = 847362;

Step 2: Initialize the interpreter.

 Allocate a tensor arena (static memory for activations), load the model, and initialize the interpreter:

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

constexpr int kTensorArenaSize = 200 * 1024;  // 200 KB

alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// Load model

const tflite::Model* model = tflite::GetModel(my_model_data);

// Register operations

tflite::MicroMutableOpResolver<10> resolver;

resolver.AddConv2D();

resolver.AddDepthwiseConv2D();

resolver.AddFullyConnected();

resolver.AddSoftmax();

// ... add all ops used by the model

// Create interpreter

tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);

// Allocate tensors

interpreter.AllocateTensors();

Step 3: Run inference.

 Copy input data to the input tensor, invoke the interpreter, and read the output tensor:

// Get input tensor

TfLiteTensor* input = interpreter.input(0);

// Copy image data to input tensor (96x96x3, int8 quantized)

for (int i = 0; i < 96*96*3; i++) {

    input->data.int8[i] = image_data[i];

}

// Run inference

TfLiteStatus invoke_status = interpreter.Invoke();

if (invoke_status != kTfLiteOk) {

    // Inference failed

}

// Get output tensor

TfLiteTensor* output = interpreter.output(0);

// Read predictions (10 classes, int8 quantized)

for (int i = 0; i < 10; i++) {

    int8_t quantized_output = output->data.int8[i];

    // Dequantize: float_output = (quantized - zero_point) * scale

    float prob = (quantized_output - output->params.zero_point) * output->params.scale;

}

Common TFLite Micro Pitfalls

Insufficient tensor arena size: If the arena is too small, AllocateTensors() fails. The error message tells you the required size—increase kTensorArenaSize and recompile.

Missing operations: If the model uses an op not registered in the OpResolver, you get a runtime error "Op not found." Add the missing op to the resolver.

Quantization mismatches: If you quantize the model with one scale/zero_point but the converter uses different values, outputs will be wrong. Always use the same representative dataset for calibration.

Memory alignment: TFLite Micro requires 16-byte alignment for the tensor arena. Use alignas(16) or ensure the array is naturally aligned.

## Edge Impulse: End-to-End Toolchain for Embedded ML

Edge Impulse is a web-based platform that provides the full deployment pipeline: data collection, model training, optimization, and deployment to embedded targets. It hides much of the toolchain complexity behind a GUI.

## Edge Impulse Workflow

Collect data: Upload training data via the web interface or collect it directly from your device using Edge Impulse's firmware.

Design impulse: Configure preprocessing (spectral features for audio, image scaling for vision) and select a model architecture (pre-built or custom).

Train: Train the model in the cloud (Edge Impulse's servers, not your machine).

Optimize: Edge Impulse automatically applies quantization.

Deploy: Download a deployment package (C++ library, Arduino library, or firmware binary) for your target hardware.

## What Edge Impulse Handles

Toolchain abstraction: You don't write conversion scripts or OpResolvers. Edge Impulse generates deployment code automatically.

Hardware optimization: Edge Impulse includes optimized kernels for ARM Cortex-M, Cortex-A, ESP32, and other platforms.

Preprocessing integration: Spectral features (MFCC, FFT) and image preprocessing are compiled into the deployment library.

## What Edge Impulse Hides

Model architecture control: You can select architectures from a menu, but you cannot define arbitrary custom layers.

Training hyperparameters: Limited control over learning rate, batch size, optimizer (compared to writing your own training script).

Debugging: If the deployed model fails, you have less visibility into what went wrong (no direct access to the TFLite converter logs or intermediate files).

When to Use Edge Impulse

Good for:

Rapid prototyping (deploy a model in hours, not weeks).

Teams without ML expertise (Edge Impulse handles training and optimization).

Standard tasks (keyword spotting, gesture recognition, image classification on common datasets).

Not ideal for:

Custom model architectures that Edge Impulse doesn't support.

Applications requiring fine-grained control over quantization or deployment.

Debugging complex deployment failures (you need access to intermediate artifacts).

## ONNX and Vendor-Specific Compilers

ONNX (Open Neural Network Exchange) is an open format for representing neural networks. It's framework-agnostic—you can train a model in PyTorch, export to ONNX, and deploy it using TensorFlow Lite, ONNX Runtime, or vendor-specific tools.

ONNX Conversion from PyTorch

import torch

import torch.onnx

# Load PyTorch model
model = torch.load('my_model.pth')

model.eval()

# Create dummy input (must match model input shape)
dummy_input = torch.randn(1, 3, 96, 96)

# Export to ONNX
torch.onnx.export(model, dummy_input, 'my_model.onnx',

                  input_names=['input'], output_names=['output'],

                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

Vendor-Specific Compilers

Many embedded AI accelerators require vendor-specific toolchains that consume ONNX or TFLite models and compile them for the accelerator.

STM32Cube.AI:

Supports TFLite, ONNX, Keras models.

Compiles for STM32 microcontrollers with or without AI accelerators.

Generates optimized C code tailored for STM32 hardware.

Vela (ARM Ethos-U):

Compiles TFLite models for ARM Ethos-U NPUs.

Applies operator splitting, memory management, and layer scheduling.

Outputs an optimized .tflite file with Ethos-U-specific metadata.

Edge TPU Compiler (Google Coral):

Compiles TFLite models for Google Coral Edge TPU.

Requires int8 quantization.

Outputs a .tflite file with TPU-specific ops.

Each compiler has quirks:

STM32Cube.AI sometimes fails on models with unsupported ops (e.g., non-standard activations). It generates a report listing supported/unsupported layers.

Vela requires all operations to be int8. Float32 layers cause compilation to fail.

Edge TPU Compiler only accelerates certain ops; others fall back to the CPU, which can make the model slower than expected.

When using vendor compilers, always check the compilation report to see which layers are accelerated and which fall back.

## Toolchain-Induced Accuracy Loss

Converters and optimizers can introduce accuracy degradation that didn't exist in the trained model. This is toolchain-induced accuracy loss—errors introduced by the deployment pipeline, not by the model itself.

Common Sources

1. Incorrect quantization calibration:

 The converter uses a calibration dataset to determine activation ranges. If the calibration set is not representative (e.g., it includes only bright images but the deployment includes dark images), quantization ranges are wrong, and accuracy drops.

Fix: Use a diverse, representative calibration dataset. Include edge cases.

2. Layer fusion side effects:

 Converters fuse layers (e.g., Conv + BatchNorm + ReLU into a single op) to improve performance. If fusion is buggy or introduces numerical instability, accuracy degrades.

Fix: Disable fusion (converter.experimental_new_converter = False in TFLite) and check if accuracy recovers.

3. Operator mismatch:

 The converter maps training ops to deployment ops. Sometimes the mapping is approximate. For example, PyTorch's F.gelu() might map to TFLite's approximate GELU, not the exact version.

Fix: Check the op mapping in the converter's documentation. Use standard ops where possible.

4. Floating-point precision loss:

 Even without quantization, float32 on a CPU can differ slightly from float32 on a GPU due to different rounding modes or SIMD optimizations.

Fix: Accept small numerical differences (<0.01% accuracy change). If differences are large (>1%), investigate.

## Measuring Toolchain-Induced Loss

Compare the converted model's accuracy to the original model's accuracy on the same validation set:

Run inference on the validation set using the original training model (Python, TensorFlow/PyTorch).

Run inference on the same validation set using the converted model (TFLite interpreter or ONNX Runtime).

Calculate accuracy for both. If they differ by >2%, toolchain introduced error.

If accuracy drops, bisect the pipeline: test after conversion (before optimization), after optimization, after compilation. Identify which step introduced the error.

## Testing and Validation After Deployment

The deployed model must produce correct results on the target hardware. Testing involves:

1. Functional Correctness: Does the Model Run?

Flash the firmware, trigger an inference, and verify it completes without crashing. Check:

No memory faults or hardfaults.

Inference returns (doesn't hang).

Output tensor has expected shape and range.

2. Numerical Correctness: Are the Outputs Correct?

Run inference on known inputs and compare outputs to expected values. For int8 quantized models, allow for quantization error (±1–2% difference is normal).

Test procedure:

Select 10–100 test inputs from the validation set.

Run inference on these inputs using the training model (Python). Record outputs.

Run inference on the same inputs on the target hardware. Record outputs.

Compare outputs element-wise. Calculate mean absolute error (MAE) and maximum error.

If MAE > 5% or max error > 10%, something is wrong (quantization mismatch, incorrect conversion, runtime bug).

3. Latency Validation: Does It Meet Timing Requirements?

Measure inference latency on the target:

Run inference 100+ times.

Record min, mean, max, and 95th percentile latency.

Verify that max latency is below the deadline.

4. Memory Validation: Does It Fit in SRAM?

Measure actual SRAM usage during inference. Check that peak usage is below capacity:

Use the tensor arena size reported by TFLite Micro.

Add stack usage (measure with a stack canary or RTOS tools).

Add heap usage (if any dynamic allocation occurs, which it shouldn't).

5. Power Validation: Does It Meet Energy Budget?

Measure current draw during inference. Integrate to get energy per inference. Verify that average power (over realistic duty cycles) is below budget.

6. Accuracy Validation: Does It Achieve Target Accuracy?

Run inference on the full validation set (if feasible—this may require serial logging or SD card storage for results) and compute accuracy. Verify it matches the training accuracy (within quantization tolerance).

## A Worked Example: Full Pipeline from Keras to Arduino Nano 33 BLE Sense

You have a keyword spotting model trained in Keras. You need to deploy it to an Arduino Nano 33 BLE Sense (nRF52840, Cortex-M4 at 64 MHz, 256 KB SRAM).

Step 1: Train (Already Done)

Model: 1D CNN with 5 convolutional layers and 2 fully connected layers. Input: 16 kHz audio, 1-second windows. Output: 10 keywords + "unknown" + "silence" (12 classes).

Validation accuracy (float32): 94.2%

Step 2: Convert to TFLite with Quantization

import tensorflow as tf

model = tf.keras.models.load_model('keyword_model.h5')

# Prepare calibration dataset (100 samples)
def representative_dataset_gen():

    for i in range(100):

        audio = load_audio_sample(i)  # Returns (1, 16000) numpy array

        yield [audio.astype(np.float32)]

# Convert
converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.representative_dataset = representative_dataset_gen

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

converter.inference_input_type = tf.int8

converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open('keyword_model.tflite', 'wb') as f:

    f.write(tflite_model)

Converted model size: 85 KB (int8).

Step 3: Test Converted Model in Python

import numpy as np

interpreter = tf.lite.Interpreter(model_content=tflite_model)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()

output_details = interpreter.get_output_details()

# Test on validation set
correct = 0

total = 0

for audio, label in validation_set:

    # Quantize input

    input_scale, input_zero_point = input_details[0]['quantization']

    audio_quantized = (audio / input_scale + input_zero_point).astype(np.int8)

    interpreter.set_tensor(input_details[0]['index'], audio_quantized)

    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])

    # Dequantize output

    output_scale, output_zero_point = output_details[0]['quantization']

    output_float = (output.astype(np.float32) - output_zero_point) * output_scale

    predicted = np.argmax(output_float)

    if predicted == label:

        correct += 1

    total += 1

accuracy = correct / total

print(f"Converted model accuracy: {accuracy * 100:.1f}%")

Result: 93.1% accuracy (1.1% loss due to quantization—acceptable).

Step 4: Integrate into Arduino Firmware

Convert .tflite to a C array using xxd:

xxd -i keyword_model.tflite > keyword_model_data.h

Write Arduino sketch:

#include <TensorFlowLite.h>
#include "keyword_model_data.h"

constexpr int kTensorArenaSize = 80 * 1024;

alignas(16) uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroInterpreter* interpreter = nullptr;

void setup() {

    Serial.begin(115200);

    // Load model

    const tflite::Model* model = tflite::GetModel(keyword_model_data);

    // Create op resolver

    static tflite::MicroMutableOpResolver<8> resolver;

    resolver.AddConv2D();

    resolver.AddFullyConnected();

    resolver.AddReshape();

    resolver.AddSoftmax();

    // ... add other ops

    // Create interpreter

    static tflite::MicroInterpreter static_interpreter(

        model, resolver, tensor_arena, kTensorArenaSize);

    interpreter = &static_interpreter;

    interpreter->AllocateTensors();

    Serial.println("Model loaded");

}

void loop() {

    // Acquire audio from microphone (not shown)

    int8_t audio_buffer[16000];

    acquire_audio(audio_buffer);

    // Copy to input tensor

    TfLiteTensor* input = interpreter->input(0);

    memcpy(input->data.int8, audio_buffer, 16000);

    // Run inference

    uint32_t start = micros();

    interpreter->Invoke();

    uint32_t latency = micros() - start;

    // Read output

    TfLiteTensor* output = interpreter->output(0);

    int predicted_class = 0;

    int8_t max_score = -128;

    for (int i = 0; i < 12; i++) {

        if (output->data.int8[i] > max_score) {

            max_score = output->data.int8[i];

            predicted_class = i;

        }

    }

    Serial.print("Keyword: ");

    Serial.print(class_names[predicted_class]);

    Serial.print(" Latency: ");

    Serial.print(latency / 1000);

    Serial.println(" ms");

    delay(1000);

}

Step 5: Deploy and Verify

Compile, flash, and test:

Measured latency: 42 ms (within 100 ms budget)

SRAM usage: 80 KB tensor arena + 20 KB firmware stack = 100 KB (fits in 256 KB)

Accuracy on-device: 92.8% (tested with 50 recorded samples played through the microphone)

The 0.3% accuracy difference between TFLite interpreter (93.1%) and on-device (92.8%) is due to microphone noise and audio preprocessing differences. Acceptable.

Deployment succeeded.

What Comes Next

You have navigated the toolchain from trained model to deployed firmware. You understand conversion, optimization, compilation, and verification. But you haven't yet integrated all the skills from this book into a complete design decision.

Chapter 14—the final chapter—synthesizes everything. You apply the full framework (constraint evaluation, model selection, optimization, and toolchain deployment) to three case studies: industrial anomaly detection, medical wearable design, and agricultural edge sensing. Each case study requires making and justifying every decision from hardware selection to model architecture to deployment strategy.

This is where theory becomes practice.


---

*[<- Chapter 12](./chapter-12-model-optimization.md) | [Table of Contents](../README.md) | [Chapter 14 ->](./chapter-14-integration-case-studies.md)*
