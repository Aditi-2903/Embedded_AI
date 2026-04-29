# Chapter 4: Inference Mechanics: What a Deployed Model Actually Does

You have a trained model. You have selected hardware. The model's parameter count fits in flash, its activation memory fits in RAM, and the theoretical inference latency calculated from FLOP count and processor throughput is within budget. You compile the model for the target, flash the firmware, and run your first test.
The inference takes three times longer than predicted. Or the output is garbage—numerical overflow or NaN values where predictions should be. Or the system crashes halfway through the first inference pass with a memory fault. The model that worked perfectly in Python on your laptop fails on the embedded target in ways you didn't anticipate.
This is not a rare failure mode. It is the default outcome when you deploy without understanding what inference actually does on constrained hardware. The theoretical analysis from Chapters 2 and 3 told you whether deployment was feasible in principle. This chapter teaches you to trace what happens in practice—how the model executes, where it fails, and how to diagnose the gap between prediction and reality.
By the end of this chapter, you will be able to trace the inference execution path on a microcontroller, identify the computational bottleneck in a model-hardware pairing, measure actual latency on target hardware, and determine whether numerical precision issues are corrupting your results. You will understand why the same model that runs smoothly on a desktop GPU requires careful optimization to run at all on an embedded processor.
The skill is not debugging in the general sense. The skill is inference-specific diagnosis: recognizing when latency violations come from cache misses versus memory bandwidth limits versus numerical precision mismatches, and knowing which fix applies to which failure.
The Inference Execution Pipeline on Bare-Metal and RTOS Systems
When you call an inference function on a microcontroller, a sequence of operations executes in a specific order. The execution pipeline determines how long inference takes, how much memory it consumes, and where failures occur. Understanding this pipeline is the prerequisite for optimization.
On a bare-metal system—no operating system, just your firmware running directly on the processor—the inference pipeline is deterministic. You allocate memory once during initialization, load the model weights from flash, and execute inference in a loop triggered by sensor data or a timer. The processor runs at a fixed clock speed with no competing processes. If inference takes 200 milliseconds on the first pass, it takes 200 milliseconds on every pass, barring cache effects or thermal throttling.
On an RTOS (real-time operating system) like FreeRTOS or Zephyr, inference runs as a task competing with other tasks for processor time. The scheduler preempts inference when higher-priority tasks need to run, the timer interrupt fires, or communication events arrive. The 200-millisecond inference might spread across 250 milliseconds of wall-clock time due to scheduling delays, or it might be interrupted entirely if a critical task preempts it.
Most embedded AI deployments use bare-metal execution for latency-critical paths and RTOS execution for systems with multiple concurrent functions. A keyword spotting system might run inference in a bare-metal loop to guarantee sub-100ms response. A wearable health monitor might run inference as a low-priority RTOS task that yields to communication and sensor acquisition.
The inference pipeline itself has four stages, regardless of whether the system is bare-metal or RTOS-based:
Stage 1: Input preparation. Sensor data arrives in raw format—ADC samples, pixel values, digital sensor readings. The data must be preprocessed into the format the model expects: normalized to the range [0,1] or [-1,1], reshaped into the correct tensor dimensions, and converted to the numeric type the model uses (float32, int8, etc.). If preprocessing includes FFT (for audio features) or image scaling, this stage can dominate total latency.
Stage 2: Inference execution. The model's computational graph executes layer by layer. Each layer loads its weights from flash (or from a cached copy in SRAM), reads its input activations, performs the layer's operation (convolution, matrix multiply, activation function), and writes the output activations to the next layer's input buffer. This is where the FLOPs are computed. This is where most of the time is spent.
Stage 3: Output interpretation. The model produces raw outputs—logits for classification, bounding boxes for detection, regression values for prediction. These must be post-processed into actionable results: applying softmax to convert logits to probabilities, applying non-maximum suppression to merge overlapping bounding boxes, or thresholding regression outputs against application-specific bounds.
Stage 4: Result delivery. The processed output is stored, transmitted, or used to trigger an action. For a keyword spotter, this might be setting a GPIO pin high. For a predictive maintenance system, it might be writing a fault code to flash or sending a LoRaWAN packet. This stage is usually negligible compared to inference itself, but network transmission or flash writes can add latency if not properly managed.
When you measure inference latency, you must decide which stages to include. The model's inference time is Stage 2 only—the time spent executing the computational graph. Application latency is Stage 1 through Stage 4—the time from sensor data arriving to action being taken. The difference can be substantial. A model with 50ms inference time might have 150ms application latency if input preprocessing includes a 256-point FFT and output delivery includes a network transmission.
Most profiling tools report model inference time (Stage 2). If your application has a hard latency requirement, you must profile the full pipeline on target hardware to verify that all stages combined meet the deadline.
Operator-Level Execution: How Operations Run on CPU vs. Specialized Hardware
A neural network is a sequence of operations: matrix multiplications, convolutions, activations, pooling. On a desktop GPU, these operations map to thousands of parallel execution units. On a microcontroller, they map to a single CPU core executing instructions sequentially, possibly assisted by SIMD extensions or a hardware accelerator.
The performance of each operation depends on how well it maps to the available hardware. Some operations are compute-bound—their execution time is determined by how many arithmetic operations must execute. Other operations are memory-bound—their execution time is determined by how fast data can move between memory and the processor.
Consider matrix multiplication: C = A × B, where A is M×K and B is K×N. The output C is M×N. The operation count is M × N × K multiply-accumulates. For a 128×128 matrix multiplied by a 128×10 matrix, that's 163,840 MACs.
On a Cortex-M4 with FPU running at 100 MHz, each MAC takes one cycle when operands are in registers. Theoretical time is 1.6 milliseconds. But operands are not in registers—they're in SRAM or flash. Loading a 128×128 matrix from SRAM requires 16,384 reads. If each read takes 2 cycles due to SRAM access latency, that's 32,768 cycles of memory overhead. The operation becomes memory-bound, and actual time increases to 3–5 milliseconds depending on cache efficiency.
Now consider the same operation on a Cortex-M7 with 16 KB of L1 data cache. If the 128×128 matrix (64 KB as float32) doesn't fit in cache, it's fetched from SRAM repeatedly. But if the algorithm is structured to multiply in 64×64 blocks, each block fits in cache and is reused across multiple output elements. Cache-aware blocking reduces memory accesses by an order of magnitude, and execution time drops to 1.8 milliseconds—close to the theoretical limit.
This is why theoretical FLOP-based latency predictions fail. Memory bandwidth, cache behavior, and data layout can change execution time by 3–10× for the same operation count.
Convolution is even more memory-intensive. A 2D convolution with a 3×3 kernel requires loading 9 input pixels for each output pixel. For a 96×96 input with 32 channels convolved to produce 32 output channels, that's 96 × 96 × 32 × 9 × 32 = 84 million memory accesses. If inputs are in external PSRAM instead of on-chip SRAM, each access costs 10–20 cycles instead of 1–2 cycles, and latency increases by an order of magnitude.
Activation functions—ReLU, sigmoid, tanh—are compute-light but can be memory-bound if not fused with the preceding operation. A ReLU applies element-wise to an activation tensor: if x < 0, output 0; else output x. For a 4096-element tensor, that's 4096 comparisons and 4096 conditional writes. On a pipelined processor, the conditional can cause branch mispredictions, which flush the pipeline and waste cycles. A well-optimized implementation uses SIMD to process 4 or 8 elements per instruction, eliminating branches entirely.
Hardware accelerators change the execution model. A neural processing unit (NPU) includes dedicated matrix multiply engines that execute hundreds of MACs per cycle. An operation that takes 100,000 cycles on a CPU might take 1,000 cycles on an NPU. But the accelerator is only faster if the operation maps to its supported primitives. A standard 3×3 convolution maps well. A depthwise convolution might map poorly if the accelerator doesn't support channel-wise operations. Attention mechanisms or dynamic control flow (if-then logic inside the network) usually don't map at all, falling back to CPU execution.
When profiling a model, you must identify which operations execute where. Most inference frameworks include a layer-wise profiling mode that reports execution time per operation. If 80% of inference time is spent in one convolutional layer, that's your bottleneck. Optimize that layer—by reducing its size, quantizing it, or offloading it to an accelerator—and total latency drops proportionally.
Memory Allocation During Inference: Static vs. Dynamic, and Why Dynamic Fails
Memory allocation strategies determine whether inference succeeds or crashes. On embedded systems, dynamic memory allocation during inference is a failure mode, not a feature.
Dynamic allocation—calling malloc() or new to allocate memory at runtime—is standard practice in desktop software. You request memory when you need it, use it, and free it when you're done. The operating system manages a heap and reclaims memory from terminated processes. If allocation fails, the system swaps to disk or terminates low-priority processes to free space.
Embedded systems have no such safety net. The heap is a fixed region of SRAM. When you allocate memory, it comes from that region. When you free memory, it returns to the heap, but fragmentation can prevent future allocations even when total free memory is sufficient. If allocation fails, there is no recovery—no swap space, no other processes to terminate. The system crashes or returns a null pointer, and inference fails.
Worse, memory allocation itself consumes time. A malloc() call searches the heap for a suitable block, which can take hundreds of microseconds. If you allocate memory during inference, that overhead adds to latency. If you allocate repeatedly inside a loop (say, allocating a temporary buffer for each convolutional layer), the overhead accumulates to milliseconds.
For embedded AI, the correct allocation strategy is static pre-allocation. All memory required for inference is allocated once during system initialization, before the first inference runs. This includes:
Model weights (loaded from flash into SRAM if needed, or accessed directly from flash)
Activation buffers for every layer's intermediate results
Scratch buffers for temporary computations (e.g., im2col transformations for convolution)
Input and output tensors
The total memory requirement is the sum of these buffers. You allocate a contiguous block of SRAM—called the tensor arena in TensorFlow Lite Micro terminology—and the inference engine manages it internally, reusing buffers where possible.
Buffer reuse is critical. A naive implementation allocates separate memory for each layer's output, which means peak memory usage is the sum of all activation tensors. A buffer-reuse implementation recognizes that once a layer's output has been consumed by the next layer, its buffer can be reused for a later layer's output. Peak memory usage becomes the size of the largest layer's output plus any buffers that must coexist.
For example, consider a three-layer network:
Layer 1 output: 4096 floats (16 KB)
Layer 2 output: 2048 floats (8 KB)
Layer 3 output: 10 floats (40 bytes)
Without buffer reuse, you need 16 + 8 + 0.04 = 24 KB. With buffer reuse, Layer 1's output buffer is reused for Layer 2's output (Layer 2 produces fewer elements, so it fits), and both are reused for Layer 3. Peak memory is 16 KB, a 33% reduction.
Static allocation also enables compile-time memory checking. If your model requires 180 KB of activation memory and your target has 200 KB of SRAM, you know at compile time that deployment is feasible (with 20 KB margin for the rest of the system). If the model requires 250 KB, you know at compile time that deployment fails, and you don't waste time debugging runtime crashes.
The workflow for static allocation is:
Profile the model on desktop to determine total activation memory requirement.
Allocate a static buffer of that size in your embedded firmware.
Initialize the inference engine with a pointer to that buffer.
Run inference. The engine uses the buffer internally and never calls malloc().
If you see malloc() calls in your inference code path, something is misconfigured. Trace them, identify the source, and replace them with pre-allocated buffers.
Latency Profiling: How to Measure and Interpret Inference Time on Target Hardware
Theoretical latency—FLOP count divided by processor throughput—is a starting estimate. Actual latency is measured on target hardware under realistic operating conditions.
The most basic profiling method is cycle counting. Most ARM Cortex-M cores include a Data Watchpoint and Trace (DWT) unit with a cycle counter. You enable the counter, read it before inference starts, read it after inference completes, and compute the difference. The result is the number of CPU cycles consumed by inference.
Here's the code pattern:
// Enable DWT cycle counter (once during initialization)
CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
DWT->CYCCNT = 0;
DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

// Measure inference
uint32_t start = DWT->CYCCNT;
run_inference();  // Your inference function
uint32_t end = DWT->CYCCNT;
uint32_t cycles = end - start;

// Convert to time
float latency_ms = (cycles / (float)CPU_FREQ_HZ) * 1000.0f;

For a processor running at 100 MHz, 10 million cycles equals 100 milliseconds. If you measure 15 million cycles, inference takes 150 milliseconds.
This tells you total latency but not where the time is spent. For operator-level profiling, you instrument each layer:
uint32_t layer1_start = DWT->CYCCNT;
run_layer1();
uint32_t layer1_end = DWT->CYCCNT;

uint32_t layer2_start = DWT->CYCCNT;
run_layer2();
uint32_t layer2_end = DWT->CYCCNT;

printf("Layer 1: %lu cycles\n", layer1_end - layer1_start);
printf("Layer 2: %lu cycles\n", layer2_end - layer2_start);

Most embedded ML frameworks include built-in profiling. TensorFlow Lite Micro has a profiler that reports per-operator timing. You enable it at compile time, run inference, and read the profiling output over UART or semihosting. The output looks like:
Operator,          Count, Time (ms)
CONV_2D,               1,      45.2
DEPTHWISE_CONV_2D,    13,      82.3
FULLY_CONNECTED,       1,       3.1
SOFTMAX,               1,       0.8
Total inference time: 131.4 ms

This immediately tells you where optimization effort should focus. If depthwise convolutions consume 63% of total time, that's your bottleneck.
Profiling must account for variance. Run inference 100 times and record the distribution of latencies. If all runs take 130–132 ms, latency is stable. If runs vary from 120 to 180 ms, you have non-determinism—possibly from cache effects, interrupt handling, or thermal throttling. For soft real-time applications, use the mean or 95th percentile latency. For hard real-time applications, use the maximum observed latency plus margin.
Energy profiling requires hardware measurement. A current sense amplifier on the device's power rail measures instantaneous current draw. You log current over time, correlate it with inference start/stop timestamps, and integrate to get energy per inference:
Energy (Joules) = ∫ V(t) × I(t) dt
For a 3.3V device drawing 40 mA during 150 ms of inference:
Energy = 3.3V × 0.040A × 0.150s = 0.0198 J = 19.8 mJ
If inference runs every 10 seconds, average power is 19.8 mJ / 10 s = 1.98 mW. A 1000 mAh battery at 3.3V stores 11,880 Joules. Battery life is 11,880 J / 1.98 mW = 6 million seconds = 69 days.
Profiling tools like SEGGER SystemView, Percepio Tracealyzer, and ARM Development Studio provide graphical timelines showing when each function executes, how long it takes, and how tasks interact on RTOS systems. These are invaluable for diagnosing scheduling delays, interrupt latency, and resource contention.
Numerical Precision: float32 vs. float16 vs. int8 and What the Tradeoffs Are
Neural networks are trained using 32-bit floating-point arithmetic. Each weight, each activation, and each intermediate result is a float32 value. This gives high precision but consumes memory and compute resources.
Embedded deployment often uses reduced precision: float16 (16-bit floating-point), bfloat16 (brain floating-point, a variant of float16), or int8 (8-bit integer). Each reduction halves memory usage and can double or quadruple throughput on processors with SIMD or specialized hardware.
But reducing precision introduces numerical errors. The errors accumulate through the network's layers, and if they grow too large, the model's predictions degrade.
float32 is the baseline. Each value ranges from ±1.4 × 10⁻⁴⁵ to ±3.4 × 10³⁸ with roughly 7 decimal digits of precision. Operations are exact within floating-point rounding error, which is negligible for neural network inference. The cost is 4 bytes per value and relatively slow execution on processors without FPUs.
float16 halves memory to 2 bytes per value. Range is ±6.5 × 10⁴ with roughly 3 decimal digits of precision. Many trained networks tolerate float16 conversion with less than 1% accuracy loss. The problem is that most microcontrollers don't support native float16 arithmetic—they emulate it in software, which is slower than float32 on an FPU. float16 is common on GPUs and some mobile processors but rare on microcontrollers.
int8 reduces memory to 1 byte per value. Integers range from -128 to 127. Neural network weights and activations don't naturally fit this range, so they must be quantized: scaled and shifted to map the float32 range onto [-128, 127]. Quantization introduces rounding error, and if not done carefully, accuracy degrades by 5–20%.
Post-training quantization (PTQ) is the simplest approach: you take a trained float32 model, measure the range of weights and activations over a representative dataset, compute scaling factors, and convert everything to int8. The inference engine multiplies int8 values, accumulates results in int32 to avoid overflow, and rescales back to int8 after each operation.
Quantization-aware training (QAT) simulates quantization during training, allowing the model to learn weights that remain accurate even after quantization. QAT typically achieves better accuracy than PTQ but requires retraining the model, which you may not control.
The compute advantage of int8 is substantial. On processors with SIMD extensions, you can pack four int8 values into a 32-bit register and perform four operations in parallel. On specialized accelerators like ARM Ethos-U55, int8 MAC throughput can be 64× higher than float32 throughput on the same chip.
The tradeoff is accuracy. For image classification, int8 quantization typically costs 1–3% accuracy. For regression tasks with small dynamic range (e.g., sensor anomaly detection), it might cost 10% or more. For some models, particularly those with batch normalization or large dynamic range in activations, int8 quantization can destroy accuracy entirely if not tuned carefully.
The decision workflow is:
Deploy the model in float32 and measure latency and memory.
If latency or memory exceeds budget, quantize to int8.
Measure accuracy loss. If loss is acceptable (<5% for most applications), deploy quantized.
If loss is unacceptable, try QAT, adjust activation ranges, or use mixed precision (some layers int8, some float32).
Never assume quantization is free. Always measure accuracy before and after, and always profile latency and memory to confirm the expected speedup materializes.
A Worked Example: Profiling Keyword Spotting on Arduino Nano 33 BLE Sense
You have a keyword spotting model trained to detect "yes," "no," and "unknown" from 1-second audio clips. The model is a depthwise separable CNN with 80,000 parameters. Theoretical analysis predicts:
Model size: 320 KB (float32)
Activation memory: 45 KB
Operation count: 22 million MACs
Target: Arduino Nano 33 BLE Sense (nRF52840, 64 MHz Cortex-M4 with FPU, 256 KB RAM, 1 MB flash)
Theoretical latency at 60 MMAC/s sustained: 22M / 60M = 367 milliseconds.
You deploy the model using TensorFlow Lite Micro, compile, flash, and run. The first inference takes 1,200 milliseconds. That's 3× slower than predicted.
Diagnosis Step 1: Check precision.
You examine the model file. It's float32. The Cortex-M4 has an FPU, so float32 should be efficient. But you check the compiled code and discover that the TFLite Micro kernel is using a reference C implementation, not the optimized CMSIS-NN kernels that leverage the FPU and DSP extensions. The reference implementation is slower by 3–5×.
Fix: Reconfigure the build to use CMSIS-NN optimized kernels. Recompile and re-flash.
New latency: 420 milliseconds. Much better—close to the 367 ms prediction. But still over budget if your application has a 100 ms deadline.
Diagnosis Step 2: Identify bottleneck.
You enable TFLite Micro's profiling output and run inference again. The profiler reports:
DEPTHWISE_CONV_2D (layer 3): 180 ms
DEPTHWISE_CONV_2D (layer 5): 120 ms
CONV_2D (layer 1): 60 ms
Other layers: 60 ms
Total: 420 ms

Two depthwise convolutions consume 71% of total time. These layers operate on 32×32 and 24×24 feature maps with 64 channels each.
Diagnosis Step 3: Check memory layout.
You inspect the activation buffers. Layers 3 and 5's inputs are in external PSRAM because total activations exceed the 256 KB on-chip SRAM budget. External memory access costs 10× more than on-chip SRAM access, which explains why these layers are so slow.
Fix: Reduce the model's activation memory by decreasing feature map sizes. You retrain the model with 16×16 maximum feature maps instead of 32×32. Activation memory drops to 28 KB, fitting entirely in on-chip SRAM.
New latency: 140 milliseconds.
Diagnosis Step 4: Quantize.
140 ms is still above the 100 ms target. You apply post-training int8 quantization. The model size drops to 80 KB, activation memory stays at 28 KB, and int8 SIMD operations execute 4× faster than float32 scalar operations on the Cortex-M4.
New latency: 45 milliseconds.
Accuracy before quantization: 94.2%
 Accuracy after quantization: 92.8%
The 1.4% accuracy loss is acceptable for the application. The model now meets all constraints: 80 KB flash, 28 KB RAM, 45 ms latency.
This is the diagnostic process. Theoretical analysis predicted 367 ms. Deployment measured 1,200 ms. Profiling identified the bottleneck (slow kernels, then external memory access). Optimization (CMSIS-NN, model shrinking, quantization) brought latency to 45 ms—well below target.
You could not have found this without measurement. The theoretical prediction was directionally correct but quantitatively wrong by 25×. Only profiling on target hardware revealed the real bottlenecks.
When Inference Fails: Common Failure Modes and How to Diagnose Them
Inference failures on embedded hardware fall into four categories: crashes, incorrect output, excessive latency, and excessive power consumption. Each has characteristic symptoms and diagnostic paths.
Crash: Memory fault or hardfault exception.
The system halts during inference, and the debugger shows a HardFault or MemManage exception. This usually means the code tried to access invalid memory—a null pointer dereference, a stack overflow, or an out-of-bounds array access.
Check activation memory allocation. If the tensor arena is too small, the inference engine may write past the end of the buffer, corrupting the stack or heap. Increase the arena size and re-test. If the crash persists, check for stack overflow—inference may be using more stack than allocated, especially if the model includes recursive operations or deep call chains.
Incorrect output: NaN, Inf, or garbage predictions.
The model runs without crashing but produces invalid results—not-a-number (NaN), infinity (Inf), or predictions that don't match expected accuracy.
Check numerical precision. If you quantized the model, verify that the quantization parameters (scale and zero-point for each layer) are correct. A mismatch between training and inference quantization ranges can produce overflow, underflow, or saturation. Re-run post-training quantization with a larger calibration dataset.
Check input preprocessing. If the model expects inputs normalized to [0,1] but you're feeding raw ADC values in the range [0,4095], the model will produce garbage. Verify that input scaling, mean subtraction, and tensor layout (NHWC vs. NCHW) match the training configuration.
Excessive latency: Inference takes 10× longer than predicted.
The model runs correctly but too slowly. Check which kernels are executing. If the framework is using reference C implementations instead of optimized SIMD or hardware-accelerated kernels, latency balloons. Reconfigure the build to enable optimized implementations.
Check memory access patterns. If activations are in external memory, latency increases by an order of magnitude. Move activations to on-chip SRAM by reducing model size or increasing SRAM allocation.
Check clock configuration. If the processor is running at 10 MHz instead of 100 MHz due to misconfigured clocks, all compute operations take 10× longer. Verify the clock tree configuration in your startup code.
Excessive power consumption: Battery drains faster than calculated.
The model runs correctly, but battery life is half of what the duty cycle calculation predicted. Measure actual current draw during inference with a current meter. If measured current is 2× higher than the datasheet active current, the model is accessing peripherals or memory in an inefficient way.
Check for busy-waiting. If the code polls a flag or spins in a loop waiting for inference to complete, the processor stays active instead of sleeping, and power consumption increases. Replace polling with interrupt-driven or DMA-driven execution.
Check for unintended peripheral activation. If the code leaves LEDs, sensors, or communication modules powered during inference, they contribute to active current. Disable all non-essential peripherals before inference and re-enable them afterward.
These diagnostic paths are not exhaustive, but they cover 90% of deployment failures. The key is systematic measurement: profile latency, measure memory usage, log output values, and compare against predictions. The gap between prediction and measurement tells you where the problem is.
What Comes Next: From Mechanics to Constraints
You can now trace inference execution, measure latency and memory usage, and identify bottlenecks. But you still haven't systematically evaluated each constraint dimension. This chapter taught you how inference works. The next six chapters teach you how to evaluate whether it works well enough—memory (Chapter 5), compute (Chapter 6), power (Chapter 7), hardware acceleration (Chapter 8), communication (Chapter 9), and real-time guarantees (Chapter 10).
Each chapter adds one diagnostic tool to your toolkit. By the end of Part II, you will be able to audit any proposed embedded AI deployment and identify every constraint violation before deployment. That audit is what separates successful integrations from expensive failures.
But first, the most visible constraint: memory. Chapter 5 examines why models that "fit" in total memory still fail at runtime, and what you can do about it.

---

*[← Chapter 3](./chapter-03-ml-for-embedded-engineers.md) | [Table of Contents](../README.md) | [Chapter 5 →](./chapter-05-memory.md)*