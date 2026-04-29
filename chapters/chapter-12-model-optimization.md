# Chapter 12: Optimizing Models: Quantization, Pruning, and Distillation

You have selected a model: MobileNetV2-0.5 achieves 89% accuracy on your validation set and was the best candidate from your evaluation. But it doesn't quite fit. The model requires 1.8 MB of flash (your target has 2 MB, but the firmware already uses 600 KB, leaving only 1.4 MB available). Activation memory is 220 KB (your target has 256 KB SRAM, but the firmware uses 80 KB, leaving 176 KB for inference). Latency is 280 milliseconds (your budget is 250 milliseconds).
Selection alone is insufficient. The model is close to viable, but not viable. You need to compress it—reduce its size, reduce its memory footprint, reduce its operation count—without destroying accuracy.
This is where model optimization comes in. Optimization techniques transform a trained model into a smaller, faster, or more efficient version that fits within deployment constraints while preserving as much accuracy as possible. The three primary techniques are quantization (reducing numeric precision), pruning (removing unnecessary weights or layers), and knowledge distillation (training a small model to mimic a large model).
This chapter teaches you to apply post-training quantization and quantization-aware training, understand the tradeoffs between different pruning strategies, use knowledge distillation to compress models beyond what pruning achieves, combine techniques in the right order, and determine the accuracy floor—when further compression breaks the application. By the end of this chapter, you will be able to take a model that barely fails constraints and transform it into a model that comfortably meets them.
Quantization: float32 → int8, What Changes, What Breaks
Quantization reduces numeric precision from floating-point (float32, float16) to fixed-point or integer (int16, int8). The primary motivation for embedded deployment is memory reduction and compute acceleration.
A float32 value occupies 4 bytes. An int8 value occupies 1 byte. Quantizing a 1 million-parameter model from float32 to int8 reduces its size from 4 MB to 1 MB—a 75% reduction. On processors with SIMD extensions, int8 operations execute 4–8× faster than float32 operations because multiple int8 values pack into a single register.
But quantization introduces numerical errors. Floating-point numbers can represent a wide dynamic range (±10⁻³⁸ to ±10³⁸) with high precision. Integers are limited to a small range (int8: -128 to 127). Mapping a float32 weight of 0.0342 to int8 requires rounding, which introduces error. When that error propagates through millions of operations, accuracy can degrade.
How Quantization Works
Quantization maps floating-point values to integers using a linear transformation:
quantized_value = round((float_value - zero_point) / scale)
where scale and zero_point are parameters computed from the float32 value distribution. The inverse operation (dequantization) recovers an approximation of the original float:
float_value_approx = (quantized_value × scale) + zero_point
For example, if weights range from -1.5 to 1.5, you set:
scale = (max - min) / 255 = 3.0 / 255 = 0.0118
zero_point = -128
A weight of 0.5 quantizes to:
q = round((0.5 - 0) / 0.0118) = round(42.4) = 42
Dequantizing gives:
f = 42 × 0.0118 + 0 = 0.495
The error is 0.005, which is small. But if you quantize an activation with range [0.001, 0.01], the scale becomes 0.01 / 255 = 0.000039, and small values near 0.001 are heavily rounded. This is where quantization damages accuracy.
Per-Tensor vs. Per-Channel Quantization
Per-tensor quantization uses a single scale and zero_point for an entire tensor (e.g., all weights in a layer). This is simple but suboptimal when weights or activations have different ranges across channels.
Per-channel quantization uses different scale and zero_point for each output channel. This better preserves accuracy when channels have imbalanced ranges (some channels have weights in [-0.1, 0.1], others in [-2.0, 2.0]).
Per-channel quantization is more accurate but requires storing N scale factors (one per channel) instead of one, which increases model size slightly. For embedded deployment, the accuracy gain usually justifies the cost.
Symmetric vs. Asymmetric Quantization
Symmetric quantization assumes the float32 range is symmetric around zero: [-α, α]. It sets zero_point = 0 and scale = 2α / 255. This simplifies math (no zero_point offset) but wastes representational range if the distribution is skewed.
Asymmetric quantization allows arbitrary ranges [min, max]. It uses both scale and zero_point, which requires more computation but better represents distributions that are not centered on zero (e.g., ReLU activations, which are always ≥0).
For embedded inference, symmetric quantization is preferred when hardware supports it (some accelerators optimize for zero_point = 0), and asymmetric otherwise.
Post-Training Quantization vs. Quantization-Aware Training
There are two paths to quantize a model: post-training quantization (PTQ) and quantization-aware training (QAT).
Post-Training Quantization (PTQ)
PTQ takes a trained float32 model and quantizes it to int8 without retraining. The process:
Run the model on a calibration dataset (100–1000 representative inputs).
Record the range of weights and activations in each layer.
Compute scale and zero_point for each tensor.
Convert weights to int8 and store them in the model file.
During inference, activations are quantized on-the-fly as they're computed.
PTQ is fast (no retraining required) and works well for many models. But it can degrade accuracy by 3–10% if:
Weights or activations have extreme outliers (one value much larger than others).
The calibration dataset is not representative.
Layers have very small weights (quantization rounds them to zero).
Quantization-Aware Training (QAT)
QAT simulates quantization during training. The model learns weights that remain accurate even after quantization. The process:
Insert fake quantization nodes into the model graph.
During forward pass, weights and activations are quantized to int8, then dequantized back to float32 for gradient computation.
Train the model normally. The optimizer adjusts weights to minimize loss under quantization.
After training, convert the model to true int8 (remove fake quantization nodes and store int8 weights).
QAT typically achieves 1–3% accuracy loss (compared to 3–10% for PTQ) but requires access to the training pipeline and dataset, which you may not have if the model came from a third party.
When to Use Each
Use PTQ when:
You don't have access to the training code or dataset.
The model already has acceptable accuracy, and 3–5% loss is tolerable.
You need a quick deployment (PTQ takes minutes, QAT takes hours or days).
Use QAT when:
Accuracy is critical, and you cannot tolerate 5%+ loss.
You control the training pipeline.
The model has many layers or complex operations that are sensitive to quantization.
Pruning: Structured vs. Unstructured, and Why Structured Matters for Embedded
Pruning removes weights from the model to reduce parameter count. There are two types: unstructured pruning and structured pruning.
Unstructured Pruning
Unstructured pruning sets individual weights to zero based on a criterion (magnitude, gradient, sensitivity). For example, you remove all weights with absolute value < 0.01. The weight matrix becomes sparse—most entries are zero, but the non-zero entries are scattered throughout the matrix.
Advantage: Maximum flexibility. You can remove 50–90% of weights with minimal accuracy loss on many models.
Disadvantage: Sparse matrices are inefficient on general-purpose CPUs and most embedded hardware. Matrix multiplication libraries (BLAS, CMSIS-NN) assume dense matrices. Sparse operations require special data structures (CSR, COO) and custom kernels, which are not available on most embedded platforms.
Result: Unstructured pruning reduces parameter count (and flash usage) but does not reduce latency or SRAM usage on embedded targets unless you implement custom sparse inference kernels, which is non-trivial.
Structured Pruning
Structured pruning removes entire structures: entire channels, entire layers, or entire filters. For example, you remove 30% of the output channels in a convolutional layer. The resulting matrix is smaller but still dense.
Advantage: Dense matrices are supported by all inference frameworks and hardware. Removing 30% of channels reduces memory, latency, and energy proportionally.
Disadvantage: Less flexibility. You must remove entire structures, which may be harder to do without accuracy loss.
Result: Structured pruning reduces all metrics (flash, SRAM, latency) and is compatible with embedded inference frameworks out-of-the-box.
For embedded deployment, structured pruning is almost always the right choice. The only exception is if you have a custom sparse inference engine or an accelerator with native sparse support (rare on embedded hardware).
How Structured Pruning Works
Train the model normally.
Measure each channel's importance (e.g., by its L1 norm, or by the accuracy drop when it's removed).
Remove the least important channels (e.g., the bottom 20% by importance).
Fine-tune the pruned model to recover accuracy.
Repeat steps 2–4 iteratively until target size is reached.
Some frameworks (TensorFlow Model Optimization Toolkit, PyTorch's torch.nn.utils.prune) automate this process.
Accuracy vs. Pruning Rate
The relationship between pruning rate and accuracy is nonlinear. Removing the first 20% of parameters typically costs <1% accuracy. Removing 40% costs 2–4%. Removing 60% costs 5–10%. Beyond 70%, accuracy collapses unless the model was over-parameterized to begin with.
The sweet spot for embedded pruning is 20–40% removal. This gives meaningful size reductions with acceptable accuracy loss. If you need more compression, combine pruning with quantization.
Knowledge Distillation: Training a Small Model to Mimic a Large Model
Knowledge distillation trains a small "student" model to replicate the behavior of a large "teacher" model. The student learns from both the ground-truth labels and the teacher's predictions, which encode richer information than labels alone.
Why Distillation Works
Consider a classification task with 10 classes. For an input image, the ground-truth label might be "cat" (class 3). A one-hot label vector is [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]. But a well-trained teacher model might output probabilities: [0.01, 0.02, 0.85, 0.05, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01].
The teacher's output is "soft"—it assigns non-zero probability to classes that are visually similar to the correct class (e.g., "dog" gets 5%). This soft distribution provides more information than the hard label. The student learns to match the teacher's distribution, which helps it generalize better than if it trained only on hard labels.
Distillation Process
Train a large, accurate teacher model (e.g., ResNet50, 90% accuracy).
Train a small student model (e.g., MobileNetV2-0.35) with a modified loss function:
L_student = α × L_hard + (1-α) × L_soft
L_hard: cross-entropy between student predictions and ground-truth labels
L_soft: cross-entropy between student predictions and teacher predictions (both softened with temperature T)
α: weighting factor (typically 0.3–0.7)
The student learns to approximate the teacher's behavior while remaining small and fast.
Temperature in Distillation
Temperature (T) is a hyperparameter that controls how "soft" the predictions are. Standard softmax is:
p_i = exp(z_i) / Σ exp(z_j)
With temperature T:
p_i = exp(z_i / T) / Σ exp(z_j / T)
Higher T (e.g., T=5 or T=10) makes the distribution softer (more uniform). Lower T (T=1) is standard softmax. During distillation, you use T > 1 to emphasize the teacher's soft probabilities.
When Distillation Helps
Distillation is effective when:
The teacher is much larger than the student (e.g., 10×+ more parameters).
The student architecture is fundamentally limited (cannot achieve teacher accuracy no matter how long you train).
You have access to the teacher model and the training data.
Distillation typically gains 2–5% accuracy over training the student from scratch. Combined with quantization and pruning, it can close the gap between a large, accurate model and a small, deployable model.
Combining Techniques: Quantize, Prune, Distill—Ordering Matters
When multiple optimization techniques are needed, the order of application affects the final result.
Recommended Order: Prune → Distill → Quantize
Step 1: Prune. Remove 20–40% of channels from the base model. Fine-tune to recover accuracy. This reduces the model's parameter count and operation count.
Step 2: Distill (optional). If the pruned model's accuracy is below threshold, train a student model that matches the pruned model's architecture but learns from a larger teacher. This recovers 2–4% accuracy.
Step 3: Quantize. Apply post-training quantization (or QAT) to convert the pruned (and optionally distilled) model to int8. This reduces memory by 75% and accelerates inference.
Why This Order?
Pruning before quantization allows quantization to operate on a smaller model, which reduces calibration complexity. Distillation after pruning uses the teacher to guide the student toward better accuracy even after channels are removed. Quantization last ensures that the final deployed model is int8, which is what the hardware accelerates.
Alternative Order: Quantize → Prune
Quantize first, then prune the quantized model. This can work but is less common because pruning int8 models is less well-supported by frameworks.
What Not to Do
Do not prune aggressively (>60%) and then quantize. Quantization adds error. Pruning adds error. Combined, they can destroy accuracy. If you need >60% compression, use distillation to a smaller architecture instead of extreme pruning.
The Accuracy Floor: When Compression Breaks the Application
Every model has an accuracy floor—the point below which further compression is unacceptable because the model no longer meets the application requirement.
The floor is application-specific:
A keyword spotter with 92% accuracy might be acceptable (8% of commands require repetition).
A medical diagnostic with 92% accuracy is unacceptable (8% misdiagnosis rate is catastrophic).
The workflow for finding the accuracy floor:
Define the minimum accuracy required by the application (from product requirements or regulatory standards).
Measure baseline accuracy (uncompressed model on validation set).
Apply compression incrementally (10%, 20%, 30% pruning; float32 → int16 → int8 quantization).
Measure accuracy after each step.
Stop compressing when accuracy drops below the floor.
Measuring Accuracy Degradation
Accuracy is not the only metric. For some applications, you care about:
False positive rate (e.g., anomaly detection—false alarms are costly)
False negative rate (e.g., pedestrian detection—missing a pedestrian is catastrophic)
Worst-case accuracy (e.g., accuracy on rare classes or edge cases, not just average)
After compression, measure all relevant metrics. A model with 90% overall accuracy might have 60% accuracy on rare classes, which could be unacceptable.
A Worked Example: Compressing a Model That Barely Fails Constraints
You are deploying MobileNetV2-0.5 for defect detection on an STM32H7 (2 MB flash, 1 MB SRAM). The model:
Parameters: 1.5 million (float32: 6 MB, int8: 1.5 MB)
Flash required: 1.8 MB (after firmware: 600 KB + model: 1.5 MB int8 + overhead: -300 KB, wait that doesn't add up)
Let me recalculate. Firmware uses 600 KB. Model weights (int8) are 1.5 MB. Total: 2.1 MB. Flash capacity: 2 MB. The model doesn't fit by 100 KB.
SRAM required: 280 KB (activations). Firmware uses 80 KB. Total: 360 KB. SRAM capacity: 1 MB. SRAM is fine.
Latency: 320 ms. Budget: 250 ms. Latency fails by 70 ms.
Accuracy (float32): 91.5%. Accuracy (int8 PTQ): 89.2%. Application requires ≥88%. Accuracy is acceptable.
Goal: Reduce flash usage by 100 KB and reduce latency by 70 ms (22% reduction) without accuracy falling below 88%.
Step 1: Apply Structured Pruning
You prune 25% of channels across all layers. The model now has 1.125 million parameters instead of 1.5 million.
New metrics:
Flash (int8): 1.125 MB
SRAM (activations): 210 KB (reduced proportionally)
Latency: 240 ms (25% reduction in operations)
Accuracy (float32, pruned): 89.8% (fine-tuned after pruning)
Analysis:
Flash: 600 KB firmware + 1.125 MB model = 1.725 MB. Fits with 275 KB margin. ✓
SRAM: 80 KB firmware + 210 KB activations = 290 KB. Fits with 734 KB margin. ✓
Latency: 240 ms < 250 ms. Fits with 10 ms margin. ✓
Accuracy: 89.8% > 88%. Acceptable. ✓
Pruning alone solved the problem. But there's a risk: you have only 10 ms of latency margin. If future firmware updates add features, latency might creep above 250 ms.
Step 2 (Optional): Distill to Recover Accuracy
The pruned model's accuracy is 89.8%, but the original float32 model achieved 91.5%. If you want to recover some of that 1.7% loss, you can train a student model with the same architecture as the pruned model but using the original large model as a teacher.
You train the student with distillation. New accuracy: 90.6%. This doesn't change latency or memory, but it gives you 0.8% more accuracy, which might be valuable if the application's accuracy threshold increases in the future.
Step 3: Quantize (Already Done)
The model is already int8. If it were still float32, you'd apply PTQ here to get the 75% memory reduction. Since you already quantized, this step is complete.
Final Model
Parameters: 1.125 million (int8)
Flash: 1.725 MB (fits in 2 MB with 14% margin)
SRAM: 290 KB (fits in 1 MB with 71% margin)
Latency: 240 ms (fits in 250 ms with 4% margin)
Accuracy: 90.6% (distilled) or 89.8% (pruned without distillation), both > 88% threshold
The model now comfortably meets all constraints. Pruning was sufficient. Distillation was optional but improved accuracy. Quantization (already applied) enabled the entire deployment.
Tools and Frameworks for Model Optimization
Several frameworks automate quantization, pruning, and distillation:
TensorFlow Model Optimization Toolkit:
PTQ: tf.lite.TFLiteConverter with optimizations=[tf.lite.Optimize.DEFAULT]
QAT: tfmot.quantization.keras.quantize_model()
Pruning: tfmot.sparsity.keras.prune_low_magnitude()
PyTorch:
PTQ: torch.quantization.quantize_dynamic() or torch.quantization.quantize()
QAT: torch.quantization.prepare_qat() → train → torch.quantization.convert()
Pruning: torch.nn.utils.prune (unstructured), or third-party libraries for structured pruning
ONNX Runtime:
Supports quantized ONNX models
Can run PTQ on ONNX graphs
Neural Network Compression Framework (NNCF):
Intel's open-source tool for quantization, pruning, and distillation
Supports TensorFlow and PyTorch
Vendor-specific tools:
STM32Cube.AI: Includes quantization and optimization for STM32 targets
Edge Impulse: Web-based tool with automated quantization and optimization
Choose based on your training framework and deployment target. If you trained in TensorFlow and deploy to TFLite Micro, use TensorFlow Model Optimization Toolkit. If you trained in PyTorch, use PyTorch quantization or convert to ONNX.
What You Cannot Fix with Optimization
Optimization reduces size, latency, and power, but it has limits. Some problems require model redesign, not compression:
Problem: Accuracy is already below threshold before optimization.
 Optimization will make it worse. You need a better base model or more training data.
Problem: The model's architecture is fundamentally wrong for the task.
 Compressing a ResNet for keyword spotting won't help. Use a 1D CNN designed for audio.
Problem: Latency is 10× over budget.
 Optimization typically gains 2–4× at best. If you need 10×, you need faster hardware or a completely different architecture.
Problem: The model uses operations unsupported by the target (e.g., attention, dynamic control flow).
 Optimization can't remove architectural dependencies. Redesign the model or upgrade the hardware.
Optimization is the final step after selection. If the selected model is fundamentally unsuitable, optimization won't save it.
What Comes Next
You can now select models (Chapter 11) and optimize them (Chapter 12). But you still haven't deployed them. Chapter 13 examines the toolchains that convert trained models into firmware—the build systems, converters, and runtime libraries that take a .h5 or .onnx file and produce a binary that runs on your microcontroller. This is where theoretical models become real deployments.

---

*[← Chapter 11](./chapter-11-model-selection.md) | [Table of Contents](../README.md) | [Chapter 13 →](./chapter-13-tinyml-toolchains.md)*