# Chapter 3 — Machine Learning for Embedded Engineers: A Functional Foundation
*The .tflite file is not a program. It's a data structure with a price tag.*

Open a .tflite file in a hex editor. You'll see a binary blob—structured data, not executable code. Somewhere in those bytes are floating-point numbers arranged in matrices, integers describing tensor shapes, metadata specifying how data flows through a graph. This file is the artifact you'll deploy to embedded hardware. It was produced by a training process you may never perform, using datasets you may never see, on servers with resources your target device cannot imagine. But once training completes, that process is irrelevant to your work. The .tflite file is what remains—a fixed data structure with known size, known operations, known resource demands.

This chapter teaches you to reason about that artifact. Not how to create it, but how to understand what it costs to execute. By the end, you'll know what happens during inference, how architectures map to memory and compute requirements, which model classes suit which sensing tasks, and how to translate machine learning terminology into the constraint language you learned in Chapter 2.

You won't learn how to train a model. You won't learn gradient descent, backpropagation, loss functions. Those are machine learning engineer concerns. Your concern is: given a trained model, can it run on constrained hardware, and if so, how much will it cost?

This is ML literacy for integration, not invention. It's enough.

<!-- → INFOGRAPHIC: Visual representation of the .tflite file structure showing the binary blob on the left with arrows pointing to extracted components on the right — weight matrices (stored in flash), tensor shapes (define activation buffers in RAM), and computational graph (determines operation count) -->

## What Training Produces

A neural network is a function approximator. You give it inputs, it produces outputs. The relationship between inputs and outputs is determined by millions of parameters learned from data. The learning process is training. The result is the model—the fixed set of parameters that define the function.

Supervised learning is the most common paradigm for embedded AI. It works like this: you collect a dataset of input-output pairs where outputs are labeled by humans or known from ground truth. For image classification, inputs are images and outputs are category labels. For keyword spotting, inputs are audio spectrograms and outputs are word labels. For anomaly detection, inputs are sensor readings and outputs are binary labels.

The training algorithm adjusts parameters to minimize the difference between predictions and true labels. This happens iteratively. The model predicts, compares to labels, computes error, updates parameters to reduce error. After thousands or millions of iterations, predictions become accurate on seen inputs. The real test is whether it generalizes—whether it produces accurate predictions on new inputs it's never encountered.

When training finishes, the model is frozen. Parameters no longer change. The model becomes a lookup table in the functional sense: given any input from the distribution it was trained on, it produces a prediction based on learned parameters. Those parameters are what you store in flash. Those parameters determine the model's size.

Training happens once, typically on a GPU server with gigabytes of RAM and hours or days of compute time. Inference—running the trained model on new inputs—happens repeatedly, on your embedded device, with the resources you quantified in Chapter 2. Training is not your problem. Inference is.

This distinction matters because the embedded literature sometimes conflates them. A paper claiming "deep learning on microcontrollers" usually means inference on microcontrollers, not training. Training a neural network on a device with 256 KB of RAM is impractical for all but the smallest models. The models you deploy were trained elsewhere. Your job is making them run efficiently in inference mode.

<!-- → IMAGE: Split diagram showing training (GPU server with gigabytes of RAM, hours of compute, iterative parameter updates) on the left, and inference (microcontroller with kilobytes of RAM, milliseconds of compute, frozen parameters) on the right, with a one-way arrow labeled "trained model" connecting them -->

## Layers, Weights, and Activations

A neural network is a directed graph. Each node is a computation, each edge carries data. In the most common architecture—the feedforward network—data flows from input to output through a sequence of layers. Each layer transforms its input and passes the result to the next. The final layer produces the model's output.

Each layer has two kinds of data: weights (also called parameters) and activations. Weights are fixed after training. Activations are computed during inference and depend on input data.

Weights are the learned parameters defining the layer's transformation. For a fully connected layer, weights are a matrix. For a convolutional layer, weights are a set of small filters. The total number of weights determines the model's memory footprint in flash. A model with 500,000 weights stored as 32-bit floats occupies 2 MB. Stored as 8-bit integers, it occupies 500 KB.

Activations are intermediate results computed as data flows through the network. When you run inference on an image, the first layer computes activations from pixels, the second layer computes activations from the first layer's output, and so on until the final layer produces the classification result. Activations live in RAM during inference. The size of activation buffers determines minimum RAM requirement.

<!-- → INFOGRAPHIC: Visual distinction between weights and activations — weights shown as fixed matrices stored in flash (non-volatile, determined at training time), activations shown as flowing tensors in RAM (volatile, computed during inference, depend on input data) -->

Consider a simple three-layer feedforward network. Input layer: 784 neurons (28×28 grayscale image). Hidden layer: 128 neurons. Output layer: 10 neurons (classification into 10 classes).

The hidden layer has weights connecting each input to each hidden neuron: 784 × 128 = 100,352 weights. As 32-bit floats, that's 401 KB. Add 128 biases, another 512 bytes. Total weight memory: ~401 KB.

The output layer has 128 × 10 = 1,280 weights plus 10 biases. Total weight memory: ~5 KB.

Total model size: ~406 KB for weights. Fits comfortably in 1 MB of flash.

But activations? During inference, you compute the hidden layer's 128 activations and store them in RAM. As 32-bit floats, that's 512 bytes. Then compute the output layer's 10 activations (40 bytes). Peak activation memory is the sum of the largest layer's output plus scratch buffers. For this simple network, under 1 KB—negligible compared to typical SRAM budgets.

This scales poorly. A convolutional network for image classification might have dozens of layers, each producing activation tensors with thousands or millions of elements. Activation memory can exceed weight memory by an order of magnitude.

## The Inference Pass

Inference is a forward pass through the network. You load an input, feed it to the first layer, compute that layer's output, feed the output to the next layer, repeat until you reach the final layer. The final layer's output is the prediction.

Each layer performs two operations: a linear transformation followed by a nonlinear activation function.

The linear transformation is a matrix multiplication (fully connected layers) or convolution (convolutional layers). This is where weights are used. For a fully connected layer:

y = W × x + b

where x is input vector, W is weight matrix, b is bias vector, y is output. The operation count is multiply-accumulate operations: (input size) × (output size). For a layer with 784 inputs and 128 outputs, that's 100,352 MACs.

The nonlinear activation function is applied element-wise to the linear transformation's output. Common ones: ReLU (rectified linear unit), sigmoid, tanh. ReLU is simplest—it replaces negative values with zero:

ReLU(y) = max(0, y)

ReLU is fast—a single comparison and conditional assignment per element. Sigmoid and tanh involve exponentials, expensive on embedded processors without hardware support.

After the activation function, the result becomes input to the next layer. Repeat until all layers execute.

<!-- → INFOGRAPHIC: Flow diagram of inference pass for three-layer network showing input (784 elements) → linear transformation W1×x+b1 (100,352 MACs) → ReLU activation → hidden layer (128 elements) → linear transformation W2×h+b2 (1,280 MACs) → softmax → output (10 class probabilities) -->

Here's the inference algorithm for a three-layer network:

1. Load input x (784-element vector from 28×28 image)
2. Compute hidden layer: h = ReLU(W1 × x + b1) (128 elements)
3. Compute output layer: y = softmax(W2 × h + b2) (10 elements)
4. Return y as predicted class probabilities

Total operation count: (784 × 128) + (128 × 10) = 101,632 MACs, plus activation function cost. On a processor sustaining 60 MMAC/s, this takes roughly 1.7 milliseconds. On a processor sustaining 10 MMAC/s (slower Cortex-M4 without FPU, using integer arithmetic), it takes 10 milliseconds.

This is inference cost: number of operations that must execute, multiplied by time per operation. The model's architecture determines operation count. The processor's throughput determines time per operation. Together, they determine latency.

## Why CNNs Dominate Embedded Vision

Fully connected layers are simple but inefficient for image data. A 224×224 RGB image has 150,528 pixels. A fully connected layer connecting that input to 128 outputs requires 19 million weights. That's 76 MB as float32—too large for most embedded targets.

Convolutional layers solve this by exploiting spatial locality. Instead of connecting every input to every output, a convolutional layer applies a small filter (3×3 or 5×5) that slides across the input, computing a weighted sum at each position. The same filter weights are reused across the entire image, dramatically reducing parameter count.

<!-- → INFOGRAPHIC: Visual comparison of fully connected layer (every input pixel connected to every output, dense connections, 19M weights) versus convolutional layer (small 3×3 filter sliding across image, sparse connections reusing same weights, 288 weights for 32 output channels) -->

A 3×3 convolutional filter applied to a single input channel has 9 weights. If the layer has 32 output channels (32 different filters), total weight count is 9 × 32 = 288 weights—orders of magnitude fewer than a fully connected layer.

Convolution also preserves spatial structure. A fully connected layer treats the image as a flat vector, discarding information about which pixels are neighbors. A convolutional layer maintains 2D structure, allowing it to detect local patterns like edges, textures, shapes.

The operation count for a convolutional layer is:

(output height) × (output width) × (output channels) × (kernel height) × (kernel width) × (input channels)

For a 224×224 input image with 3 channels (RGB), a 3×3 convolution producing 32 output channels:

224 × 224 × 32 × 3 × 3 × 3 = 43.6 million MACs

Expensive—but still far cheaper than the equivalent fully connected layer.

Convolutional networks (CNNs) for image classification typically consist of several convolutional layers to extract features, pooling layers to reduce spatial resolution, and a few fully connected layers at the end to combine features and produce class predictions.

CNNs dominate embedded vision—face detection, object recognition, defect classification—because they achieve high accuracy with manageable parameter counts. A well-designed CNN for 96×96 image classification might have 200,000–500,000 parameters, fitting comfortably in 1–2 MB of flash when quantized.

## MobileNets: Efficient Architectures for Embedded

Standard convolutional layers are expensive because they combine spatial filtering and channel mixing in a single operation. A 3×3 convolution with 32 input channels and 64 output channels performs 3 × 3 × 32 × 64 = 18,432 multiplications per output pixel.

Depthwise separable convolution splits this into two cheaper operations: a depthwise convolution that filters each input channel independently, followed by a pointwise convolution (1×1) that mixes channels.

For the same example:

Depthwise: 3 × 3 × 32 = 288 multiplications per output pixel
Pointwise: 1 × 1 × 32 × 64 = 2,048 multiplications per output pixel
Total: 2,336 multiplications per output pixel

That's 8 times fewer operations than standard convolution, with minimal accuracy loss for many tasks.

<!-- → INFOGRAPHIC: Side-by-side comparison showing standard convolution (single operation combining spatial filtering and channel mixing, 18,432 multiplications) versus depthwise separable convolution (two-stage operation: depthwise 288 multiplications + pointwise 2,048 multiplications = 2,336 total, 8× reduction) -->

MobileNet is a family of CNN architectures built entirely from depthwise separable convolutions. MobileNetV1 was the first widely-used efficient architecture for embedded vision. MobileNetV2 added inverted residual blocks and linear bottlenecks, improving accuracy while maintaining low cost. MobileNetV3 used neural architecture search to optimize for specific hardware targets.

A typical MobileNetV2 model for 224×224 image classification has 3.5 million parameters (~14 MB float32, 3.5 MB int8), 300 million MACs per inference, and 90% top-5 accuracy on ImageNet.

For embedded deployment, MobileNets are often scaled down. A 96×96 input resolution with 0.35× width multiplier (reducing channels in each layer) gives 400,000 parameters (~1.6 MB float32, 400 KB int8), 25 million MACs per inference, and application-specific accuracy.

This scaled-down version fits on an STM32H7 (2 MB flash, 1 MB RAM) and runs inference in under 200 ms. It's the sweet spot for embedded vision: accurate enough for real-world tasks, cheap enough to deploy on microcontrollers.

## Why RNNs Are Hard to Deploy

Recurrent neural networks (RNNs) are designed for sequence data—time series, audio, text. Unlike feedforward networks that process each input independently, RNNs maintain a hidden state that carries information from previous time steps.

The hidden state makes RNNs powerful for tasks like speech recognition and anomaly detection in sensor streams. But it also makes them difficult to deploy on constrained hardware because the hidden state must be stored in RAM between time steps, and state size scales with the model's capacity.

<!-- → INFOGRAPHIC: Comparison showing feedforward network (no state between inferences, each input processed independently) versus RNN (hidden state maintained in RAM between time steps, state size: 128 units = 512 bytes for simple RNN, 2 KB for LSTM, updated 100 times per second for audio) -->

A simple RNN layer with 128 hidden units requires storing 128 activations between time steps. LSTM (long short-term memory) requires four times as much state—512 values for a 128-unit LSTM layer. For a model processing audio at 16 kHz with 10 ms frames, that state is updated 100 times per second, and every update involves reading and writing the hidden state to RAM.

RNNs also have high operation counts. Each time step performs a matrix multiplication involving the input and hidden state. For a 128-unit LSTM processing 40-dimensional audio features, the operation count is roughly 260,000 MACs per time step. Over a 1-second audio window (100 frames), that's 26 million MACs—comparable to a small CNN.

The combination of state memory requirements and high operation counts makes RNNs the most challenging model class for embedded deployment. When possible, embedded audio applications prefer alternative architectures like 1D convolutional networks or depthwise separable convolutions, which achieve similar accuracy with lower cost.

## When Neural Networks Are Overkill

Not every embedded AI application needs a neural network. For structured data—sensor readings with known features like temperature, pressure, vibration frequency—decision trees and random forests often achieve equivalent accuracy with far lower cost.

A decision tree is a series of if-then rules. Each node tests a feature against a threshold and branches left or right. The leaves contain class labels or regression values. Inference is a single path from root to leaf, requiring one comparison per tree level.

A decision tree with 10 levels and 1,023 nodes (complete binary tree) requires 1,023 threshold values stored in flash—4 KB as float32. Inference performs 10 comparisons—negligible compute cost, even on a Cortex-M0. No activation buffers needed. No floating-point arithmetic required if thresholds are quantized to integers.

<!-- → INFOGRAPHIC: Decision tree visualization showing binary tree structure with 10 levels, inference path highlighted from root to leaf requiring only 10 comparisons, with comparison to neural network showing orders of magnitude difference in memory (4 KB vs 400 KB) and compute (10 comparisons vs millions of MACs) -->

Random forests are ensembles of decision trees. Each tree votes on the output, and the majority vote determines the final prediction. A random forest with 50 trees, each 10 levels deep, requires 200 KB of flash (50 trees × 4 KB) and performs 500 comparisons per inference. Still cheaper than most neural networks.

For applications like anomaly detection in industrial sensors, predictive maintenance, or binary classification tasks with engineered features, decision trees should be the default choice. Only when accuracy gains from deep learning justify the cost should you move to neural networks.

The trade-off: decision trees are interpretable, fast, and small, but require good feature engineering. Neural networks can learn features from raw data (pixels, audio samples) but cost more in memory and compute. If your data is already structured and you know which features matter, use a decision tree. If your data is raw and high-dimensional, use a neural network.

## The Three Metrics That Determine Cost

Every model has three resource metrics you must quantify before deployment:

**Parameter count** is the total number of weights and biases. This determines the model's size in flash. A model with 1 million parameters stored as float32 requires 4 MB. Stored as int8, it requires 1 MB. Parameter count is architecture-dependent—deeper networks and wider layers have more parameters.

**FLOP count** (floating-point operations) is the number of arithmetic operations required per inference. This determines compute cost and latency. For embedded deployment, MAC count (multiply-accumulate operations) is often used instead because it maps directly to processor throughput.

**Activation memory** is the size of intermediate tensors during inference. This determines minimum RAM requirement. Activation memory depends on the largest layer's output size and whether the inference engine reuses buffers or allocates new memory for each layer.

<!-- → TABLE: The three cost metrics showing metric name, what it determines, typical embedded constraint, and example calculation — rows for parameter count (flash requirement, 1 MB available, 500k params × 2 bytes = 1 MB), FLOP/MAC count (compute latency, 60 MMAC/s throughput, 42M MACs ÷ 60 MMAC/s = 700 ms), activation memory (RAM requirement, 256 KB available, 180 KB activations + 30 KB stack = 210 KB) -->

These three metrics are not independent. Reducing parameter count (by pruning layers) also reduces FLOP count and usually reduces activation memory. But the relationship isn't always linear. A model with 50% fewer parameters might have only 30% fewer FLOPs if the pruned layers were small and remaining layers are large.

To get these metrics for a trained model, use the framework's profiling tools. TensorFlow has tf.profiler or tflite.Analyzer. PyTorch has torchinfo.summary() or thop.profile(). The output lists each layer's parameter count, output shape, and operation count. Sum them to get total model cost.

For example, TensorFlow Lite's analyzer might report:

- Model size: 1.2 MB
- Total parameters: 315,000
- Total MACs: 42 million
- Activation memory: 180 KB

Now you have the numbers you need. Go back to Chapter 2's constraint analysis:

- Does 1.2 MB fit in flash? (Check flash capacity)
- Does 180 KB fit in RAM? (Check SRAM capacity minus system overhead)
- Can the processor execute 42 million MACs within the latency budget? (Check throughput)

If all three constraints are satisfied, the model is deployable. If any constraint fails, you must either select different hardware or modify the model.

## The Translation Table

Machine learning terminology and embedded systems terminology describe the same underlying resources in different languages. Here's the translation:

**Model size** = Flash memory requirement. 1.2 MB model needs ≥1.2 MB free flash.

**Parameter count** = Weight memory footprint. 300k params × 4 bytes = 1.2 MB.

**Activation memory** = RAM requirement for inference. 180 KB activations needs ≥180 KB free SRAM.

**FLOP count / MAC count** = Compute operations per inference. 42M MACs @ 60 MMAC/s = 700 ms latency.

**Inference latency** = Execution time per forward pass. Time from input loaded to output ready.

**Batch size** = Number of inputs processed together. Always 1 on embedded (no batching).

**Quantization** = Reducing numeric precision. float32 → int8 saves 75% memory.

**Pruning** = Removing weights/layers. Reduces param count and FLOPs.

**Input shape** = Sensor data dimensions. 96×96×3 image = 27,648 input values.

**Output shape** = Prediction dimensions. 10 classes = 10-element output vector.

<!-- → TABLE: ML-to-embedded translation table with two columns showing ML term on left and embedded equivalent with example on right, covering all the translations listed above -->

When a machine learning engineer says "the model has high memory footprint," they mean it has many parameters and large activations. When you say "it doesn't fit in RAM," you mean activation memory exceeds available SRAM. You're describing the same problem in different terms.

Learning this translation lets you communicate across domains. When you collaborate with ML engineers to select or optimize a model, you can specify constraints in their language ("we need activation memory under 150 KB") and they can propose solutions in theirs ("try MobileNetV2 with 0.35 width multiplier").

## Which Model Class for Which Task

Not every model architecture is appropriate for every embedded sensing task. The model class you choose depends on input data structure, application accuracy requirements, and available hardware resources.

**For image classification** (camera-based sensing, visual inspection, object detection): Use convolutional neural networks. Start with MobileNetV2 or MobileNetV3 as baseline. Scale down input resolution and width multiplier until you meet memory and compute constraints.

**For keyword spotting and audio classification** (wake word detection, sound event recognition): Use 1D convolutional networks on audio spectrograms or depthwise separable convolutions on raw waveforms. Avoid RNNs unless you have sufficient RAM for hidden state and sufficient compute for recurrent operations.

**For time series anomaly detection** (vibration monitoring, predictive maintenance, health monitoring): Start with decision trees or random forests if you can engineer good features (statistical moments, frequency-domain features, autocorrelation). If raw sensor streams are required, use 1D CNNs or autoencoders.

**For structured tabular data** (sensor fusion, multi-modal classification with engineered features): Use decision trees, random forests, or gradient-boosted trees. Neural networks rarely outperform tree-based models on small structured datasets, and trees are far cheaper to deploy.

**For gesture recognition and motion classification** (accelerometer-based interfaces, activity tracking): Use 1D CNNs or LSTMs if you have enough RAM. For ultra-low-power devices, use decision trees on hand-crafted features like peak acceleration, zero-crossing rate, and signal energy.

<!-- → TABLE: Model selection guide showing task type, recommended model class, typical resource requirements (flash, RAM, MACs), and example application for each category: image classification, audio classification, time series anomaly detection, structured tabular data, and gesture recognition -->

The pattern: use the simplest model that meets accuracy requirements. Decision trees are simpler than neural networks. Feedforward networks are simpler than recurrent networks. Small CNNs are simpler than large CNNs. Every increment in model complexity costs memory, compute, and power. Only increase complexity when the accuracy gain justifies the cost.

## What You Still Cannot Do

You can now read a model's specification and determine its resource cost. You can identify which model class is appropriate for a given sensing task. You can translate between ML terminology and embedded constraints. But you cannot yet answer: "Can this model run on this hardware?"

That question requires integrating model cost (which you now understand) with hardware constraints (which Chapter 2 taught you to quantify). It requires knowing how to measure actual inference latency, not just theoretical operation count. It requires understanding how quantization, pruning, and hardware accelerators change the cost profile. And it requires knowing when theoretical analysis is insufficient and you must profile on real hardware.

Those skills are built in Part II. Chapter 4 begins by teaching you to measure and optimize inference mechanics—the actual execution of the model on the target processor. But before you can measure, you must understand what you're measuring. That understanding is what this chapter provided.

The .tflite file is no longer a binary blob. It's a data structure with known size, known operations, and known memory requirements. It's a design artifact you can evaluate against hardware constraints before you deploy it.

That evaluation is what comes next.
