# Chapter 11: Selecting Models for Constrained Deployment

You have evaluated a MobileNetV2 model for on-device image classification. The model achieves 89% accuracy on your validation set, fits in 1.2 MB of flash, requires 180 KB of activation memory, and runs in 220 milliseconds on your target hardware. All constraints are satisfied—barely. Flash usage is 60% of capacity, SRAM usage is 70%, and latency is 20 milliseconds under the 240 millisecond deadline.

Then your colleague sends you a different model: EfficientNet-Lite0. It achieves 91% accuracy on the same validation set, occupies 1.8 MB of flash, requires 240 KB of activation memory, and runs in 280 milliseconds. It's more accurate, but it violates the flash constraint, the SRAM constraint, and the latency constraint.

You have a third option: a custom depthwise separable CNN designed specifically for your dataset. It achieves 87% accuracy, occupies 600 KB of flash, requires 90 KB of activation memory, and runs in 110 milliseconds. It's less accurate than both alternatives but comfortably meets all constraints with margin.

Which model do you choose? The answer depends on which constraints are binding and which accuracy threshold is mandatory. If accuracy must be ≥88%, the custom model fails. If flash must be ≤1.5 MB, EfficientNet-Lite0 fails. If all constraints are hard ceilings, only MobileNetV2 survives—but it has no margin for future feature additions or model updates.

This is the model selection problem. Until now, the model was given, and you evaluated feasibility. In Part III, the model is chosen. Model selection is the first step in model-aware design: evaluating candidates not on ImageNet accuracy but on deployment metrics—latency on target hardware, memory footprint, energy per inference, and accuracy under quantization.

This chapter teaches you to compare models on constraint profiles, use deployment-aware benchmarks instead of generic accuracy benchmarks, understand what makes efficient architectures efficient, and navigate the accuracy-efficiency Pareto frontier to select the model that best fits your application. By the end of this chapter, you will be able to justify a model selection against hardware constraints and explain when a less accurate model is the right choice.

## Why Accuracy Benchmarks Are the Wrong Metric for Embedded Selection

The machine learning research community evaluates models on standardized benchmarks: ImageNet top-1 accuracy for image classification, COCO mAP (mean average precision) for object detection, WER (word error rate) for speech recognition. Papers are published, models are ranked, and the highest-accuracy model "wins."

But benchmark accuracy is a poor proxy for embedded suitability. A model with 95% ImageNet accuracy and 80 million parameters is useless on a microcontroller with 1 MB of flash. A model with 90% accuracy and 5 million parameters might deploy successfully.

Benchmark datasets are also mismatched to deployment conditions. ImageNet contains 1.2 million high-resolution images of 1,000 object categories, photographed in diverse settings with professional equipment. Your deployment might involve 96×96 grayscale images of 10 categories, captured by a low-cost camera in a factory. Training on ImageNet and deploying on factory images introduces a domain gap that degrades accuracy by 5–20%.

## The relevant metrics for embedded model selection are:

Latency on target hardware: Not FLOPs (which predict CPU latency poorly) or ImageNet inference time (measured on a GPU), but measured milliseconds on your actual embedded processor running your actual inference framework.

Memory footprint: Not parameter count alone but the sum of weight memory (flash) and activation memory (SRAM) for your target's quantization precision (int8, not float32).

Energy per inference: Not theoretical operations but measured millijoules on target hardware under realistic duty cycles.

Accuracy on your dataset: Not ImageNet accuracy but validation accuracy on data that matches your deployment distribution. A model with 95% ImageNet accuracy might have 70% accuracy on your application-specific dataset if the domain gap is large.

Accuracy under quantization: Many models that tolerate int8 quantization with <1% accuracy loss on ImageNet degrade by 10–20% on smaller, domain-specific datasets where quantization noise is more disruptive.

## The model selection workflow is:

Define deployment constraints (target hardware, latency budget, memory budget, power budget, accuracy threshold).

Identify candidate models that might fit (based on architecture surveys, prior work, model zoos).

For each candidate, measure latency, memory, energy, and accuracy on target hardware with target data.

Eliminate candidates that violate hard constraints.

Among remaining candidates, select the one that best balances accuracy, margin, and future extensibility.

This is empirical and iterative. You cannot determine the best model from a datasheet or a paper. You must profile each candidate on your hardware with your data.

## Deployment-Aware Metrics

Deployment-aware metrics quantify what matters for embedded integration. Here are the five core metrics and how to measure them.

1. Latency on Target Hardware

Measure end-to-end inference time on the actual deployment target, not a development board with different clock speed or memory configuration. Use the actual inference framework (TensorFlow Lite Micro, not TensorFlow Lite for desktop). Include all preprocessing (normalization, resizing) and postprocessing (softmax, NMS) if they're part of the deployed pipeline.

Run inference 100+ times on diverse inputs. Report mean, 95th percentile, and maximum observed latency. The maximum is your empirical WCET estimate.

Example measurement:

Model: MobileNetV2-0.35 (scaled down)

Target: STM32H7 at 480 MHz, CMSIS-NN kernels

Input: 96×96×3 RGB images

Measured latency: mean 185 ms, 95th percentile 192 ms, max 198 ms

2. Memory Footprint (Flash + SRAM)

Measure actual deployment memory usage, not theoretical:

Flash: Compile the model into the firmware and check the linker map file for total flash usage (firmware + model weights). Subtract baseline firmware size to isolate model contribution.

SRAM: Measure tensor arena size (reported by TFLite Micro's interpreter) plus any additional buffers for preprocessing or postprocessing.

Example measurement:

Model: MobileNetV2-0.35, int8 quantized

Flash (weights): 420 KB

SRAM (activations): 140 KB

Total memory: 560 KB

3. Energy per Inference

Measure on target hardware with a current sense resistor and oscilloscope or power profiler. Log current draw over the full inference cycle (preprocessing, inference, postprocessing). Integrate to get total energy.

Energy (J) = V × ∫ I(t) dt

For battery-powered systems, also measure average power over realistic duty cycles (e.g., inference every 10 seconds, sleep between).

Example measurement:

Model: MobileNetV2-0.35

Target: nRF52840 at 64 MHz

Supply voltage: 3.3 V

Active current during inference: 45 mA

Inference time: 350 ms

Energy per inference: 3.3 V × 0.045 A × 0.35 s = 0.052 J = 52 mJ

4. Accuracy on Deployment Dataset

Train or fine-tune the model on data that matches the deployment distribution. If your application is defect detection on stamped metal parts, train on images of stamped metal parts—not ImageNet, not COCO. If you must use a pretrained model, fine-tune the final layers on your dataset.

Evaluate accuracy on a held-out validation set that matches deployment conditions: same lighting, same camera, same resolution, same noise characteristics.

## Example measurement:

Model: MobileNetV2-0.35, fine-tuned on factory defect dataset (5,000 images, 3 classes)

Validation accuracy (float32): 91.2%

Validation accuracy (int8 post-training quantization): 89.8%

Accuracy drop due to quantization: 1.4%

5. Accuracy Under Quantization

Never assume quantization is free. Measure accuracy before and after quantization on your validation set. If the drop exceeds 3–5%, investigate:

Is the calibration dataset representative? (PTQ needs diverse inputs to estimate activation ranges)

Are there layers with very small weights? (Quantization can destroy near-zero weights)

Are there layers with large dynamic range? (Activations that vary from 0.001 to 100 are hard to quantize to int8)

If PTQ accuracy is unacceptable, try quantization-aware training (QAT) or mixed precision (some layers int8, some float32).

## Efficient Architecture Families: What Made Them Efficient

Not all neural network architectures are equally suitable for embedded deployment. Some architectures were designed explicitly for efficiency—to maximize accuracy per FLOP, per parameter, or per watt. Understanding what makes these architectures efficient helps you select or design models for constrained targets.

MobileNet (v1, v2, v3)

Key innovation: Depthwise separable convolution replaces standard convolution. A standard 3×3 convolution with C input channels and C output channels requires 9C² multiplications per output pixel. A depthwise separable convolution splits this into:

Depthwise: 9C multiplications (filter each channel independently)

Pointwise: C² multiplications (1×1 convolution to mix channels)

Total: 9C + C² ≈ C² for large C (factor of 9× reduction)

Why it's efficient: Fewer operations, smaller models, lower latency. MobileNetV2 adds inverted residual blocks and linear bottlenecks to improve accuracy without increasing cost. MobileNetV3 uses neural architecture search to optimize for mobile CPUs and includes Squeeze-and-Excitation blocks for better feature representation.

Embedded suitability: Excellent. MobileNets scale with a width multiplier (0.25, 0.35, 0.5, 0.75, 1.0) that reduces channel count, and a resolution multiplier that reduces input size. You can shrink a MobileNet to fit almost any memory or latency budget at the cost of accuracy.

EfficientNet-Lite

Key innovation: Compound scaling—instead of scaling depth (more layers), width (more channels), or resolution (larger images) independently, EfficientNet scales all three together using a formula derived from neural architecture search.

Why it's efficient: Balances model capacity across dimensions. Adding layers without adding width creates a bottleneck. Adding width without depth limits expressiveness. EfficientNet finds the optimal ratio.

EfficientNet-Lite is a simplified version of EfficientNet designed for edge deployment: removes Squeeze-and-Excitation blocks (which are expensive on CPUs) and uses ReLU6 instead of Swish (simpler activation).

Embedded suitability: Good for edge processors (Raspberry Pi, Jetson Nano) but marginal for microcontrollers. Even EfficientNet-Lite0 (the smallest variant) has 4.6 million parameters and 0.4 GFLOPS, which is too large for most MCUs.

SqueezeNet

Key innovation: Fire modules that "squeeze" channels with 1×1 convolutions (reducing feature map size), then "expand" with a mix of 1×1 and 3×3 convolutions. This reduces parameters dramatically while maintaining accuracy.

Why it's efficient: Fewer parameters (SqueezeNet achieves AlexNet-level accuracy with 50× fewer parameters). But operation count is not proportionally reduced, so latency improvements are smaller than parameter reductions suggest.

Embedded suitability: Good for memory-constrained MCUs but not compute-constrained ones. SqueezeNet trades parameters for operations, which is useful when flash is the bottleneck but not when CPU throughput is.

## YOLO-Nano and Tiny-YOLO

Key innovation: Stripped-down versions of the YOLO (You Only Look Once) object detection family, designed for embedded vision. Tiny-YOLO has 9 convolutional layers instead of 50+, uses smaller feature maps, and reduces anchor box count.

Why it's efficient: Object detection is inherently expensive (must classify and localize multiple objects per frame). Tiny-YOLO makes aggressive tradeoffs—lower resolution (416×416 instead of 608×608), fewer layers, fewer anchor boxes—to enable real-time inference on edge devices.

Embedded suitability: Marginal for MCUs, good for edge processors. Tiny-YOLOv3 still requires ~15 million parameters and 5.5 GFLOPS, which exceeds most MCU budgets but runs at 20–30 FPS on a Raspberry Pi 4.

Decision Trees and Random Forests (for structured data)

Key innovation: Not neural networks—tree-based models that make sequential decisions based on feature thresholds. Inference is a walk from root to leaf, requiring one comparison per tree level.

Why it's efficient: No matrix multiplications, no floating-point arithmetic (if thresholds are quantized to integers), and deterministic execution paths. A decision tree with 10 levels requires 10 comparisons per inference—trivial compute cost.

Embedded suitability: Excellent for structured data (sensor features, tabular data). Terrible for raw high-dimensional data (images, audio) where feature extraction requires convolutions.

When to use: If you can engineer good features from sensor data (statistical moments, frequency-domain features, correlation coefficients), decision trees outperform neural networks on small datasets and constrained hardware.

## Task-Specific Model Selection

Different tasks have different requirements. The optimal architecture for image classification is not optimal for keyword spotting or anomaly detection.

Image Classification (96×96 or smaller)

Candidate architectures: MobileNetV2 (0.35 or 0.5 width multiplier), MobileNetV3-Small, SqueezeNet, custom depthwise separable CNN.

Selection criteria: Latency dominates. Memory is secondary (models are small at this resolution). Accuracy depends on dataset complexity (simple datasets: 3 classes, custom CNN suffices; complex datasets: 10+ classes, MobileNet required).

Baseline model: MobileNetV2-0.35 with 96×96 input. ~400K parameters, 25M MACs, 88–92% accuracy on typical embedded vision tasks.

Keyword Spotting (Audio)

Candidate architectures: Depthwise separable CNN on spectrograms (DS-CNN), 1D CNN on raw audio, small LSTM, decision tree on engineered features.

Selection criteria: SRAM is the bottleneck (RNNs have large state). Prefer feedforward architectures. If features are engineered (MFCCs, zero-crossing rate), decision trees may outperform neural networks.

Baseline model: DS-CNN with 12 layers, ~100K parameters, <5 ms latency on Cortex-M4, 95%+ accuracy on Google Speech Commands dataset.

## Time-Series Anomaly Detection

Candidate architectures: Decision trees on statistical features, 1D CNN on raw sequences, autoencoder (if SRAM allows), LSTM (if SRAM and compute allow).

Selection criteria: Depends on whether you have labeled anomalies (supervised) or only normal data (unsupervised). Supervised: decision trees or 1D CNN. Unsupervised: autoencoder (trains on normal data, flags high reconstruction error as anomaly).

Baseline model: Random forest with 50 trees on 20 engineered features (mean, variance, FFT peaks, autocorrelation). ~200 KB model size, <1 ms inference, 85–90% anomaly detection rate.

Gesture Recognition (Accelerometer)

Candidate architectures: 1D CNN, decision tree on features, small LSTM.

Selection criteria: Depends on gesture complexity. Simple gestures (tap, shake, rotate): decision tree. Complex temporal patterns (drawing shapes in the air): 1D CNN or LSTM.

Baseline model: 1D CNN with 4 convolutional layers, 64K parameters, 8M MACs, 90%+ accuracy on typical gesture datasets.

## The Accuracy-Efficiency Pareto Frontier

For any task, there exists a set of models that are Pareto-optimal: no other model is both more accurate and more efficient. The Pareto frontier is the boundary of this set, where improving accuracy requires sacrificing efficiency (latency, memory, or energy).

Plotting the Pareto frontier helps visualize the tradeoff space. On a graph with accuracy (y-axis) and latency (x-axis), Pareto-optimal models lie on the upper-left frontier. Models below and to the right are dominated—there's a better model that's both faster and more accurate.

Example: Face detection models evaluated on a custom dataset

Model

Latency (ms)

Accuracy (%)

Memory (KB)

TinyFace (custom)

45

84

120

MobileNetV2-0.25

80

88

250

MobileNetV2-0.35

110

91

420

MobileNetV3-Small

95

90

380

EfficientNet-Lite0

180

94

1,800

ResNet18

450

96

11,000

Plotting accuracy vs. latency:

TinyFace: 45 ms, 84% → Pareto-optimal (fastest, acceptable accuracy)

MobileNetV2-0.25: 80 ms, 88% → dominated by MobileNetV3-Small (95 ms, 90% is better)

MobileNetV3-Small: 95 ms, 90% → Pareto-optimal

MobileNetV2-0.35: 110 ms, 91% → Pareto-optimal (slightly slower but more accurate than MobileNetV3-Small)

EfficientNet-Lite0: 180 ms, 94% → Pareto-optimal (much slower but more accurate)

ResNet18: 450 ms, 96% → dominated by EfficientNet-Lite0 (180 ms, 94% is nearly as accurate and 2.5× faster)

The Pareto frontier consists of TinyFace, MobileNetV3-Small, MobileNetV2-0.35, and EfficientNet-Lite0. Which model you choose depends on your constraints:

If latency < 50 ms is mandatory → TinyFace (only option)

If latency < 100 ms and accuracy ≥ 90% → MobileNetV3-Small

If latency < 120 ms and accuracy ≥ 91% → MobileNetV2-0.35

If latency < 200 ms and accuracy ≥ 94% → EfficientNet-Lite0

The key insight is that accuracy and efficiency are not independent. You cannot have the highest accuracy and the lowest latency. You must choose a point on the Pareto frontier based on which constraint is binding.

## Neural Architecture Search (NAS) for Embedded Targets

Neural Architecture Search (NAS) automates the process of finding efficient architectures. Instead of hand-designing a network, you specify:

A search space (possible layers, layer configurations, connections)

An objective function (e.g., maximize accuracy subject to latency < 100 ms on target hardware)

A search algorithm (reinforcement learning, evolutionary algorithms, gradient-based)

NAS explores the search space, evaluates candidates on the objective function, and converges to an architecture optimized for your constraints.

## Examples of NAS-generated models:

MobileNetV3: Used NAS to optimize for mobile CPUs. The search algorithm tuned layer widths, kernel sizes, and activation functions to maximize accuracy per FLOP.

EfficientNet: Used NAS to discover the compound scaling rule (how to scale depth, width, and resolution together).

ProxylessNAS: Designed to minimize latency on mobile phones, running the search directly on target hardware (phones, not GPUs).

Once-for-All (OFA): Trains a single "super-network" that can be sliced into sub-networks of different sizes, allowing you to extract models optimized for specific latency or memory budgets without retraining.

When to use NAS:

High-volume products where a 1% accuracy improvement or 10% latency reduction justifies the search cost (GPU-days or GPU-weeks).

Tasks where hand-designed architectures are suboptimal (e.g., new sensor modalities, unusual input dimensions).

Applications with unusual constraints (e.g., latency < 20 ms on a specific NPU).

When NOT to use NAS:

Low-volume or one-off deployments (the search cost exceeds the value of the optimized model).

When a well-known architecture (MobileNet, EfficientNet) already meets constraints.

When you lack the compute resources to run NAS (requires GPU access and ML expertise).

For most embedded AI projects, using an existing efficient architecture (MobileNet, SqueezeNet) scaled to your constraints is more practical than running NAS. NAS is a research tool that occasionally produces better models than human designers, but it's not necessary for successful deployment.

## A Worked Example: Selecting a Model for Wearable Fall Detection

You are designing a wearable fall detection system. The device uses a 3-axis accelerometer sampled at 100 Hz. Falls must be detected within 500 ms of occurrence to trigger an alert. The device runs on a coin cell battery that must last 6 months.

Constraints:

Target hardware: nRF52840 (Cortex-M4 at 64 MHz, 256 KB SRAM, 1 MB flash)

Latency: ≤500 ms from sample window complete to decision

Power: <2 mA average (6 months on 220 mAh coin cell)

Accuracy: ≥95% fall detection rate, ≤1% false positive rate

You evaluate five candidate models:

Model A: Threshold Detector (Classical Algorithm)

Description: Hand-tuned threshold on peak acceleration and impact duration. No machine learning.

Performance:

Latency: <1 ms

Memory: 1 KB (code only, no model)

Power: negligible (simple comparisons)

Accuracy: 78% fall detection, 8% false positives

Analysis: Meets latency and power constraints but fails accuracy requirement by a wide margin. False positive rate is unacceptable (8 false alarms per 100 normal movements).

## Model B: Decision Tree on Engineered Features

Description: Random forest (50 trees) trained on 15 features extracted from 1-second accelerometer windows (mean, variance, peak, zero-crossing rate, FFT dominant frequency, etc.).

Performance:

Latency: 3 ms (feature extraction) + 0.5 ms (tree traversal) = 3.5 ms

Memory: 200 KB (tree structure)

Power: 2 ms active at 15 mA = 0.03 mJ per inference

Accuracy: 92% fall detection, 2% false positives

Analysis: Meets latency and power constraints. Memory is tight (200 KB / 256 KB SRAM leaves only 56 KB for firmware). Accuracy is below the 95% threshold.

Model C: 1D CNN (Small)

Description: 4 convolutional layers + 2 fully connected layers. Input: 100 samples × 3 axes (300 values). Parameters: 45,000 (int8).

Performance:

Latency: 25 ms (measured on nRF52840 with CMSIS-NN)

Memory: 45 KB (weights) + 30 KB (activations) = 75 KB

Power: 25 ms active at 15 mA = 0.38 mJ per inference

Accuracy: 96% fall detection, 0.8% false positives

Analysis: Meets all constraints. Latency well within 500 ms budget. Memory comfortable (75 KB / 256 KB SRAM). Power acceptable (if inference runs every 1 second, average power is 0.38 mJ / 1 s = 0.38 mW = 0.12 mA, well below 2 mA budget). Accuracy exceeds threshold. This is a viable option.

Model D: 1D CNN (Large)

Description: 6 convolutional layers + 3 fully connected layers. Parameters: 180,000 (int8).

Performance:

Latency: 85 ms

Memory: 180 KB (weights) + 80 KB (activations) = 260 KB

Power: 85 ms active at 15 mA = 1.28 mJ per inference

Accuracy: 98% fall detection, 0.3% false positives

Analysis: Fails SRAM constraint (260 KB > 256 KB). Even if you added external PSRAM, the power cost increases due to external memory access. Not viable.

## Model E: LSTM

Description: 2-layer LSTM (32 units per layer) + 1 fully connected layer. Parameters: 60,000 (int8).

Performance:

Latency: 120 ms

Memory: 60 KB (weights) + 90 KB (activations + state) = 150 KB

Power: 120 ms active at 15 mA = 1.8 mJ per inference

Accuracy: 97.5% fall detection, 0.5% false positives

Analysis: Meets all constraints. Slightly better accuracy than Model C but 5× slower and 2.4× higher power consumption. Viable but Model C is more efficient.

Comparison Summary

Model

Latency

Memory

Accuracy (Fall / FP)

Meets Constraints?

A: Threshold

1 ms

1 KB

78% / 8%

No (accuracy)

B: Random Forest

3.5 ms

200 KB

92% / 2%

No (accuracy)

C: 1D CNN (Small)

25 ms

75 KB

96% / 0.8%

Yes

D: 1D CNN (Large)

85 ms

260 KB

98% / 0.3%

No (memory)

E: LSTM

120 ms

150 KB

97.5% / 0.5%

Yes (but slower)

Decision: Select Model C (1D CNN Small). It meets all constraints with margin, uses 29% of SRAM (leaving 71% for firmware), and achieves 96% fall detection accuracy—above the 95% threshold. Model E is viable but unnecessarily slow and power-hungry. Model D is not viable due to memory constraints.

The selection process required profiling all five candidates on target hardware with the actual dataset. Model D looked promising on paper (highest accuracy) but failed in practice (exceeded SRAM). Model A worked on previous-generation devices (simpler hardware, lower expectations) but no longer meets modern accuracy standards.

This is model-aware design. You didn't just evaluate whether the first model fit. You compared multiple candidates, measured their deployment metrics, and selected the one that best balanced accuracy, constraints, and margin.

## What Comes Next

You can now select models for deployment constraints. But the model you select may not fit perfectly—it might be 10% too large, 20% too slow, or 3% below the accuracy threshold. Chapter 12 teaches you to modify models through quantization, pruning, and knowledge distillation to make them fit when selection alone is insufficient.


---

*[<- Chapter 10](./chapter-10-real-time-ai.md) | [Table of Contents](../README.md) | [Chapter 12 ->](./chapter-12-model-optimization.md)*
