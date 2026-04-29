# Chapter 3: Machine Learning for Embedded Engineers: A Functional Foundation

Open a .tflite file in a hex editor. You'll see a binary blob—structured data, not executable code. Somewhere in those bytes are floating-point numbers arranged in matrices, integers describing tensor shapes, and metadata specifying how data flows through a computational graph. This file is the artifact you will deploy to embedded hardware. It was produced by a training process you may never perform yourself, using datasets you may never see, on servers with resources your target device cannot imagine. But once training completes, that process is irrelevant to your work. The .tflite file is what remains—a fixed data structure with known size, known operations, and known resource demands.
This chapter teaches you to reason about that artifact. Not how to create it, but how to understand what it costs to execute. By the end of this chapter, you will know what happens during inference, how neural network architectures map to memory and compute requirements, which model classes are appropriate for which embedded sensing tasks, and how to translate machine learning terminology into the constraint language you learned in Chapter 2.
You will not learn how to train a model. You will not learn gradient descent, backpropagation, or loss functions. Those are the concerns of machine learning engineers. Your concern is: given a trained model, can it run on constrained hardware, and if so, how much will it cost?
This is ML literacy for integration, not for invention. It is enough.
Supervised Learning: The Process That Produces the Artifact
A neural network is a function approximator. You give it inputs, it produces outputs, and the relationship between inputs and outputs is determined by millions of parameters learned from data. The learning process is called training. The result of training is the model—the fixed set of parameters that define the function.
Supervised learning is the most common training paradigm for embedded AI applications. It works like this: you collect a dataset of input-output pairs where the outputs are labeled by humans or known from ground truth. For image classification, the inputs are images and the outputs are category labels ("cat," "dog," "car"). For keyword spotting, the inputs are audio spectrograms and the outputs are word labels ("yes," "no," "unknown"). For anomaly detection, the inputs are sensor readings and the outputs are binary labels ("normal," "fault").
The training algorithm adjusts the model's parameters to minimize the difference between the model's predictions and the true labels. This happens iteratively: the model makes predictions, compares them to the labels, computes an error, and updates the parameters to reduce that error. After thousands or millions of iterations over the dataset, the model's predictions become accurate on inputs it has seen. The real test is whether it generalizes—whether it produces accurate predictions on new inputs it has never encountered.
When training finishes, the model is frozen. The parameters no longer change. The model becomes a lookup table in the functional sense: given any input from the distribution it was trained on, it produces a prediction based on the learned parameters. Those parameters are what you store in flash memory. Those parameters are what determine the model's size.
Training happens once, typically on a GPU server with gigabytes of RAM and hours or days of compute time. Inference—running the trained model on new inputs—happens repeatedly, on your embedded device, with the resources you quantified in Chapter 2. Training is not your problem. Inference is.
This distinction is critical because the embedded systems literature sometimes conflates the two. A paper that claims "deep learning on microcontrollers" usually means inference on microcontrollers, not training. Training a neural network on a device with 256 KB of RAM is impractical for all but the smallest models and datasets. The models you deploy were trained elsewhere. Your job is to make them run efficiently in inference mode.
Neural Network Anatomy: Layers, Weights, and Activations
A neural network is a directed graph where each node is a computation and each edge carries data. In the most common architecture—the feedforward network—data flows from input to output through a sequence of layers. Each layer performs a transformation on its input and passes the result to the next layer. The final layer produces the model's output.
Each layer has two kinds of data: weights (also called parameters) and activations. Weights are fixed after training. Activations are computed during inference and depend on the input data.
Weights are the learned parameters that define the layer's transformation. For a fully connected layer, the weights are a matrix. For a convolutional layer, the weights are a set of small filters (kernels). The total number of weights in the network determines its memory footprint in flash. A model with 500,000 weights stored as 32-bit floats occupies 2 MB. Stored as 8-bit integers, it occupies 500 KB.
Activations are the intermediate results computed as data flows through the network. When you run inference on an image, the first layer computes activations from the image pixels, the second layer computes activations from the first layer's output, and so on until the final layer produces the classification result. Activations live in RAM during inference. The size of the activation buffers determines the minimum RAM requirement.
Consider a simple feedforward network with three layers:
Input layer: 784 neurons (28×28 grayscale image)
Hidden layer: 128 neurons
Output layer: 10 neurons (classification into 10 classes)
The hidden layer has weights connecting each input neuron to each hidden neuron: 784 × 128 = 100,352 weights. As 32-bit floats, that's 401 KB. The hidden layer also has 128 biases (one per neuron), adding another 512 bytes. Total weight memory for this layer: ~401 KB.
The output layer has 128 × 10 = 1,280 weights plus 10 biases. Total weight memory: ~5 KB.
Total model size: ~406 KB for weights. This fits comfortably in 1 MB of flash.
But what about activations? During inference, you compute the hidden layer's 128 activations and store them in RAM. As 32-bit floats, that's 512 bytes. Then you compute the output layer's 10 activations (40 bytes). The peak activation memory is the sum of the largest layer's output plus any scratch buffers needed for computation. For this simple network, that's under 1 KB—negligible compared to typical SRAM budgets.
This scales poorly. A convolutional network for image classification might have dozens of layers, each producing activation tensors with thousands or millions of elements. The activation memory requirement can exceed the weight memory requirement by an order of magnitude.
The Inference Pass: What Happens When Input Enters the Model
Inference is a forward pass through the network. You load an input (an image, an audio spectrogram, a sensor reading), feed it to the first layer, compute that layer's output, feed the output to the next layer, and repeat until you reach the final layer. The final layer's output is the model's prediction.
Each layer performs two operations: a linear transformation followed by a nonlinear activation function.
The linear transformation is a matrix multiplication (for fully connected layers) or a convolution (for convolutional layers). This is where the weights are used. For a fully connected layer, the transformation is:
y = W × x + b
where x is the input vector, W is the weight matrix, b is the bias vector, and y is the output. The operation count is the number of multiply-accumulate (MAC) operations: (input size) × (output size). For a layer with 784 inputs and 128 outputs, that's 100,352 MACs.
The nonlinear activation function is applied element-wise to the output of the linear transformation. Common activation functions include ReLU (rectified linear unit), sigmoid, and tanh. ReLU is the simplest: it replaces negative values with zero. Mathematically:
ReLU(y) = max(0, y)
ReLU is fast—it's a single comparison and conditional assignment per element. Sigmoid and tanh involve exponentials, which are more expensive on embedded processors without hardware support.
After the activation function, the result becomes the input to the next layer. The process repeats until all layers have executed.
Here's the inference algorithm for a three-layer network:
Load input x (e.g., 784-element vector from a 28×28 image)
Compute hidden layer: h = ReLU(W1 × x + b1) (128 elements)
Compute output layer: y = softmax(W2 × h + b2) (10 elements)
Return y as the predicted class probabilities
The total operation count is (784 × 128) + (128 × 10) = 101,632 MACs, plus the cost of activation functions. On a processor sustaining 60 MMAC/s, this takes roughly 1.7 milliseconds. On a processor sustaining 10 MMAC/s (a slower Cortex-M4 without FPU, using integer arithmetic), it takes 10 milliseconds.
This is the inference cost: the number of operations that must execute, multiplied by the time per operation. The model's architecture determines the operation count. The processor's throughput determines the time per operation. Together, they determine latency.
Convolutional Neural Networks: Why They Dominate Embedded Vision
Fully connected layers are simple but inefficient for image data. A 224×224 RGB image has 150,528 pixels. A fully connected layer connecting that input to 128 outputs requires 19 million weights. That's 76 MB as float32—too large for most embedded targets.
Convolutional layers solve this by exploiting spatial locality. Instead of connecting every input to every output, a convolutional layer applies a small filter (e.g., 3×3 or 5×5) that slides across the input, computing a weighted sum at each position. The same filter weights are reused across the entire image, which dramatically reduces the parameter count.
A 3×3 convolutional filter applied to a single input channel has 9 weights. If the layer has 32 output channels (32 different filters), the total weight count is 9 × 32 = 288 weights—orders of magnitude fewer than a fully connected layer.
Convolution also preserves spatial structure. A fully connected layer treats the image as a flat vector, discarding information about which pixels are neighbors. A convolutional layer maintains the 2D structure, allowing it to detect local patterns like edges, textures, and shapes.
The operation count for a convolutional layer is:
(output height) × (output width) × (output channels) × (kernel height) × (kernel width) × (input channels)
For a 224×224 input image with 3 channels (RGB), a 3×3 convolution producing 32 output channels:
224 × 224 × 32 × 3 × 3 × 3 = 43.6 million MACs
That's expensive—but still far cheaper than the equivalent fully connected layer.
Convolutional networks (CNNs) for image classification typically consist of:
Several convolutional layers to extract features
Pooling layers to reduce spatial resolution (e.g., max pooling, which takes the maximum value in each 2×2 region)
A few fully connected layers at the end to combine features and produce class predictions
CNNs dominate embedded vision tasks—face detection, object recognition, defect classification—because they achieve high accuracy with manageable parameter counts. A well-designed CNN for 96×96 image classification might have 200,000–500,000 parameters, fitting comfortably in 1–2 MB of flash when quantized.
Recurrent Neural Networks: Why They're Hard to Deploy
Recurrent neural networks (RNNs) are designed for sequence data—time series, audio, text. Unlike feedforward networks, which process each input independently, RNNs maintain a hidden state that carries information from previous time steps.
The hidden state makes RNNs powerful for tasks like speech recognition and anomaly detection in sensor streams. But it also makes them difficult to deploy on constrained hardware because the hidden state must be stored in RAM between time steps, and the state size scales with the model's capacity.
A simple RNN layer with 128 hidden units requires storing 128 activations between time steps. A more powerful variant called LSTM (long short-term memory) requires four times as much state—512 values for a 128-unit LSTM layer. For a model processing audio at 16 kHz with 10 ms frames, that state is updated 100 times per second, and every update involves reading and writing the hidden state to RAM.
RNNs also have high operation counts. Each time step performs a matrix multiplication involving the input and the hidden state. For a 128-unit LSTM processing 40-dimensional audio features, the operation count is roughly 260,000 MACs per time step. Over a 1-second audio window (100 frames), that's 26 million MACs—comparable to a small CNN.
The combination of state memory requirements and high operation counts makes RNNs the most challenging model class for embedded deployment. When possible, embedded audio applications prefer alternative architectures like 1D convolutional networks or depthwise separable convolutions, which achieve similar accuracy with lower cost.
MobileNets and Depthwise Separable Convolutions: Efficient Architectures for Embedded
Standard convolutional layers are expensive because they combine spatial filtering and channel mixing in a single operation. A 3×3 convolution with 32 input channels and 64 output channels performs 3 × 3 × 32 × 64 = 18,432 multiplications per output pixel.
Depthwise separable convolution splits this into two cheaper operations: a depthwise convolution that filters each input channel independently, followed by a pointwise convolution (1×1) that mixes channels.
For the same example:
Depthwise: 3 × 3 × 32 = 288 multiplications per output pixel
Pointwise: 1 × 1 × 32 × 64 = 2,048 multiplications per output pixel
Total: 2,336 multiplications per output pixel
That's 8 times fewer operations than standard convolution, with minimal accuracy loss for many tasks.
MobileNet is a family of CNN architectures built entirely from depthwise separable convolutions. MobileNetV1 was the first widely-used efficient architecture for embedded vision. MobileNetV2 added inverted residual blocks and linear bottlenecks, improving accuracy while maintaining low cost. MobileNetV3 used neural architecture search to optimize for specific hardware targets.
A typical MobileNetV2 model for 224×224 image classification has:
3.5 million parameters (~14 MB as float32, 3.5 MB as int8)
300 million MACs per inference
Top-5 accuracy of 90% on ImageNet
For embedded deployment, MobileNets are often scaled down. A 96×96 input resolution with 0.35× width multiplier (reducing the number of channels in each layer) gives:
400,000 parameters (~1.6 MB as float32, 400 KB as int8)
25 million MACs per inference
Application-specific accuracy (depends on the task and dataset)
This scaled-down version fits on an STM32H7 (2 MB flash, 1 MB RAM) and runs inference in under 200 ms. It's the sweet spot for embedded vision: accurate enough for real-world tasks, cheap enough to deploy on microcontrollers.
Decision Trees and Random Forests: When Neural Networks Are Overkill
Not every embedded AI application needs a neural network. For structured data—sensor readings with known features like temperature, pressure, vibration frequency—decision trees and random forests often achieve equivalent accuracy with far lower cost.
A decision tree is a series of if-then rules. Each node tests a feature against a threshold and branches left or right. The leaves contain class labels or regression values. Inference is a single path from root to leaf, requiring one comparison per tree level.
A decision tree with 10 levels and 1,023 nodes (a complete binary tree) requires 1,023 threshold values stored in flash—4 KB as float32. Inference performs 10 comparisons—negligible compute cost, even on a Cortex-M0. No activation buffers are needed. No floating-point arithmetic is required if thresholds are quantized to integers.
Random forests are ensembles of decision trees. Each tree votes on the output, and the majority vote determines the final prediction. A random forest with 50 trees, each 10 levels deep, requires 200 KB of flash (50 trees × 4 KB) and performs 500 comparisons per inference. That's still cheaper than most neural networks.
For applications like anomaly detection in industrial sensors, predictive maintenance, or binary classification tasks with engineered features, decision trees should be the default choice. Only when accuracy gains from deep learning justify the cost should you move to neural networks.
The decision tree vs. neural network trade-off is this: decision trees are interpretable, fast, and small, but they require good feature engineering. Neural networks can learn features from raw data (pixels, audio samples) but cost more in memory and compute. If your data is already structured and you know which features matter, use a decision tree. If your data is raw and high-dimensional, use a neural network.
Model Size, Parameter Count, and FLOP Count: The Three Metrics That Determine Cost
Every model has three resource metrics you must quantify before deployment:
Parameter count is the total number of weights and biases in the model. This determines the model's size in flash. A model with 1 million parameters stored as float32 requires 4 MB. Stored as int8, it requires 1 MB. Parameter count is architecture-dependent—deeper networks and wider layers have more parameters.
FLOP count (floating-point operations) is the number of arithmetic operations required per inference. This determines compute cost and latency. FLOP count includes multiplications, additions, and sometimes activations, depending on how it's measured. For embedded deployment, MAC count (multiply-accumulate operations) is often used instead because it maps directly to processor throughput.
Activation memory is the size of the intermediate tensors during inference. This determines the minimum RAM requirement. Activation memory depends on the largest layer's output size and whether the inference engine reuses buffers or allocates new memory for each layer.
These three metrics are not independent. Reducing parameter count (e.g., by pruning layers) also reduces FLOP count and usually reduces activation memory. But the relationship is not always linear. A model with 50% fewer parameters might have only 30% fewer FLOPs if the pruned layers were small and the remaining layers are large.
To get these metrics for a trained model, use the framework's profiling tools:
TensorFlow: tf.profiler or tflite.Analyzer
PyTorch: torchinfo.summary() or thop.profile()
ONNX: onnx.helper.printable_graph()
The output will list each layer's parameter count, output shape, and operation count. Sum them to get the total model cost.
For example, TensorFlow Lite's analyzer might report:
Model size: 1.2 MB
Total parameters: 315,000
Total MACs: 42 million
Activation memory: 180 KB

Now you have the numbers you need. Go back to Chapter 2's constraint analysis:
Does 1.2 MB fit in flash? (Check flash capacity)
Does 180 KB fit in RAM? (Check SRAM capacity minus system overhead)
Can the processor execute 42 million MACs within the latency budget? (Check throughput)
If all three constraints are satisfied, the model is deployable. If any constraint fails, you must either select different hardware or modify the model.
The ML-to-Embedded Translation Table
Machine learning terminology and embedded systems terminology describe the same underlying resources in different languages. This table translates between them:
ML Term
Embedded Equivalent
Example
Model size
Flash memory requirement
1.2 MB model → needs ≥1.2 MB free flash
Parameter count
Weight memory footprint
300k params × 4 bytes = 1.2 MB
Activation memory
RAM requirement for inference
180 KB activations → needs ≥180 KB free SRAM
FLOP count / MAC count
Compute operations per inference
42M MACs @ 60 MMAC/s = 700 ms latency
Inference latency
Execution time per forward pass
Time from input loaded to output ready
Batch size
Number of inputs processed together
Always 1 on embedded (no batching)
Quantization
Reducing numeric precision
float32 → int8 saves 75% memory
Pruning
Removing weights/layers
Reduces param count and FLOPs
Input shape
Sensor data dimensions
96×96×3 image = 27,648 input values
Output shape
Prediction dimensions
10 classes = 10-element output vector

When a machine learning engineer says "the model has high memory footprint," they mean it has many parameters and large activations. When you say "it doesn't fit in RAM," you mean the activation memory exceeds available SRAM. You're describing the same problem in different terms.
Learning this translation allows you to communicate across domains. When you collaborate with ML engineers to select or optimize a model, you can specify constraints in their language ("we need activation memory under 150 KB") and they can propose solutions in theirs ("try MobileNetV2 with 0.35 width multiplier").
Which Model Class for Which Task
Not every model architecture is appropriate for every embedded sensing task. The model class you choose depends on the input data structure, the application's accuracy requirements, and the available hardware resources.
For image classification (camera-based sensing, visual inspection, object detection):
 Use convolutional neural networks. Start with MobileNetV2 or MobileNetV3 as the baseline. Scale down the input resolution and width multiplier until you meet memory and compute constraints. If accuracy is still insufficient, try EfficientNet-Lite, which uses neural architecture search to optimize for efficiency.
For keyword spotting and audio classification (wake word detection, sound event recognition):
 Use 1D convolutional networks on audio spectrograms or depthwise separable convolutions on raw waveforms. Avoid RNNs unless you have sufficient RAM for hidden state and sufficient compute for recurrent operations. Models like DS-CNN (depthwise separable CNN) achieve 95%+ accuracy on keyword spotting with under 100 KB of weights.
For time series anomaly detection (vibration monitoring, predictive maintenance, health monitoring):
 Start with decision trees or random forests if you can engineer good features (statistical moments, frequency-domain features, autocorrelation). If raw sensor streams are required, use 1D CNNs or autoencoders. Autoencoders learn a compressed representation of normal behavior and flag deviations as anomalies.
For structured tabular data (sensor fusion, multi-modal classification with engineered features):
 Use decision trees, random forests, or gradient-boosted trees. Neural networks rarely outperform tree-based models on small structured datasets, and trees are far cheaper to deploy.
For gesture recognition and motion classification (accelerometer-based interfaces, activity tracking):
 Use 1D CNNs or LSTMs if you have enough RAM. For ultra-low-power devices, use decision trees on hand-crafted features like peak acceleration, zero-crossing rate, and signal energy.
The pattern is this: use the simplest model that meets accuracy requirements. Decision trees are simpler than neural networks. Feedforward networks are simpler than recurrent networks. Small CNNs are simpler than large CNNs. Every increment in model complexity costs memory, compute, and power. Only increase complexity when the accuracy gain justifies the cost.
What You Still Cannot Do
You can now read a model's specification and determine its resource cost. You can identify which model class is appropriate for a given sensing task. You can translate between ML terminology and embedded constraints. But you cannot yet answer the question: "Can this model run on this hardware?"
That question requires integrating the model cost (which you now understand) with the hardware constraints (which Chapter 2 taught you to quantify). It requires knowing how to measure actual inference latency, not just theoretical operation count. It requires understanding how quantization, pruning, and hardware accelerators change the cost profile. And it requires knowing when the theoretical analysis is insufficient and you must profile on real hardware.
Those skills are built in Part II. Chapter 4 begins by teaching you to measure and optimize inference mechanics—the actual execution of the model on the target processor. But before you can measure, you must understand what you're measuring. That understanding is what this chapter provided.
The .tflite file is no longer a binary blob. It is a data structure with known size, known operations, and known memory requirements. It is a design artifact that you can evaluate against hardware constraints before you deploy it.
That evaluation is what Chapter 4 teaches next.

---

*[← Chapter 2](./chapter-02-embedded-constraints-as-design-variables.md) | [Table of Contents](../README.md) | [Chapter 4 →](./chapter-04-inference-mechanics.md)*