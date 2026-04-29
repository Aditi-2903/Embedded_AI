# Chapter 5: Memory: Fitting Models into Constrained Spaces

The datasheet says 256 KB of RAM. Your model's weights occupy 180 KB. Simple arithmetic: 256 - 180 = 76 KB of headroom. The model fits.
You compile, flash, and run. The system crashes 40 milliseconds into the first inference with a memory fault. The debugger shows the heap allocator failed, malloc returned null, and the inference engine tried to write to address 0x00000000.
The model's weights were never the problem. Weights live in flash. What crashed was activation memory—the intermediate tensors created during inference, which live in RAM. Your model's activation footprint is 210 KB. Add 30 KB for the Bluetooth stack, 8 KB for the system heap, 4 KB for the call stack, and you need 252 KB before the model even starts. The model that "fit" by parameter count doesn't fit at runtime.
This failure mode is common because activation memory is invisible until you measure it. Datasheets specify total RAM. Model summaries report parameter count and weights size. Neither tells you how much RAM inference will consume while executing. That number—activation memory—is the hidden constraint that determines whether a model deploys successfully or crashes the first time it runs.
This chapter teaches you to calculate activation memory before deployment, identify which components of a model consume the most memory and why, apply memory layout strategies to reduce footprint, and determine whether a model fits within a target's memory envelope with enough margin for the rest of the system. By the end of this chapter, you will never deploy a model without first accounting for every byte of RAM it requires.
Flash vs. SRAM in AI Deployment: Where Weights Live, Where Activations Live
Memory in embedded systems is partitioned into regions with different properties. The most critical distinction for AI deployment is flash versus SRAM.
Flash is non-volatile storage. Data written to flash persists when power is removed. Flash is abundant—microcontrollers commonly have 512 KB to 2 MB of flash. But flash is read-only during normal operation. You cannot write to flash from running code without invoking special erase-and-program cycles that take milliseconds and degrade the flash over time (limited to ~10,000–100,000 erase cycles).
For AI deployment, flash stores:
The firmware binary (your application code, the inference engine, libraries)
The model's weights and biases (trained parameters that don't change during inference)
Constant lookup tables (e.g., quantization parameters, activation function tables)
SRAM is volatile storage. Data in SRAM is lost when power is removed. SRAM is fast—reads and writes complete in 1–2 CPU cycles. SRAM is writable—you can modify it freely without wear concerns. But SRAM is scarce—microcontrollers commonly have 64 KB to 512 KB of SRAM, and high-end parts top out at 1–2 MB.
For AI deployment, SRAM stores:
The application's heap and stack
Global and static variables
RTOS task control blocks and message queues (if using an RTOS)
The model's activation tensors during inference
Scratch buffers for temporary computations
The memory constraint is not "does the model fit in total memory?" but "does each component fit in the right type of memory?"
Consider a model with 500,000 parameters (2 MB as float32, 500 KB as int8) deployed on an STM32L4 with 640 KB of SRAM and 2 MB of flash. The int8 weights fit in flash with room to spare. But if the model's activations require 400 KB of SRAM, and the application already uses 200 KB for the Bluetooth stack and buffers, total SRAM usage is 600 KB—within the 640 KB limit, but with only 40 KB of margin. If the system also needs a 64 KB DMA buffer for sensor data, SRAM usage exceeds capacity, and deployment fails.
Some microcontrollers support execute-in-place (XIP) from flash, allowing the processor to fetch instructions and read-only data directly from flash without copying to SRAM. This is efficient for weights—the inference engine reads weights from flash as needed during each layer's computation. But activations cannot live in flash because they change during inference. Activations must be in writable memory, which means SRAM or external DRAM.
External memory (PSRAM, SDRAM, or eMMC) expands total memory capacity but at a cost. External memory is slower than on-chip SRAM—access latency increases from 1–2 cycles to 10–50 cycles depending on the interface. External memory consumes more power because data traverses the external bus. And external memory increases BOM cost and board complexity.
If your model's activations exceed on-chip SRAM, you have four options:
Reduce the model's activation footprint (shrink layers, reduce channels, use depthwise separable convolutions)
Add external PSRAM and accept the latency and power penalty
Stream activations to flash between layers (extremely slow, only viable for ultra-low-latency-tolerant applications)
Reject the target and select hardware with more SRAM
The decision depends on whether the latency and power cost of external memory is acceptable for the application. For a battery-powered wearable with a 100 ms latency budget, external memory might violate both power and latency constraints. For a mains-powered industrial sensor with a 1-second latency budget, external memory might be perfectly acceptable.
Weight Memory: Parameter Count × Precision = Storage Requirement
The size of a model's weights is straightforward to calculate. Count the parameters, multiply by bytes per parameter, and you have the flash requirement.
For a fully connected layer with input_size inputs and output_size outputs: Parameters = (input_size × output_size) + output_size
The first term is the weight matrix. The second term is the bias vector (one bias per output).
For a 2D convolutional layer with input_channels, output_channels, and a kernel_height × kernel_width filter: Parameters = (kernel_height × kernel_width × input_channels × output_channels) + output_channels
For a depthwise separable convolution (depthwise + pointwise): Depthwise params = (kernel_height × kernel_width × channels) + channels
 Pointwise params = (channels × output_channels) + output_channels
 Total = depthwise + pointwise
Most ML frameworks report total parameter count when you compile the model. TensorFlow's model.summary() lists parameters per layer. PyTorch's torchinfo.summary() does the same. If you don't have access to the framework, you can calculate manually from the architecture specification.
Precision determines bytes per parameter:
float32: 4 bytes
float16: 2 bytes
int8: 1 byte
A model with 1 million parameters occupies:
4 MB as float32
2 MB as float16
1 MB as int8
Quantization from float32 to int8 reduces weight memory by 75%. For a device with 1 MB of flash and 200 KB already consumed by firmware, a 1.2 MB float32 model doesn't fit, but the 300 KB int8 version does.
Weight memory is fixed after training. During inference, weights are read-only. They never change. This makes weight memory the easiest constraint to satisfy—if the weights don't fit in flash, you quantize them or prune the model until they do.
The one complication is that some inference engines copy weights from flash to SRAM for faster access. TensorFlow Lite Micro, for example, can load weights into the tensor arena if configured to do so. This speeds up inference (SRAM access is faster than flash access) but consumes SRAM. If your SRAM budget is tight, configure the engine to access weights directly from flash and accept the latency penalty.
Activation Memory: The Hidden Footprint
Activation memory is the RAM consumed by intermediate tensors during inference. When you run a forward pass through the network, each layer produces an output tensor that becomes the input to the next layer. Those tensors exist in memory while the layer executes. The total size of those tensors—plus any scratch buffers the inference engine uses internally—is the activation memory footprint.
Unlike weight memory, activation memory is not obvious from the model architecture. You cannot calculate it by counting parameters. You must trace the dataflow through the network and account for every intermediate tensor that exists simultaneously.
Consider a simple convolutional network:
Input: 96×96×3 image (27,648 elements, 110 KB as float32)
Conv1: 96×96×32 output (294,912 elements, 1.15 MB as float32)
Pool1: 48×48×32 output (73,728 elements, 288 KB as float32)
Conv2: 48×48×64 output (147,456 elements, 576 KB as float32)
Pool2: 24×24×64 output (36,864 elements, 144 KB as float32)
Flatten and FC: 10 outputs (10 elements, 40 bytes as float32)
If the inference engine allocates separate buffers for each layer's output, peak memory usage is the sum of all activation tensors: 110 + 1,150 + 288 + 576 + 144 + 0.04 = 2,268 KB. That's 2.2 MB of SRAM, which exceeds the capacity of most microcontrollers.
But the engine doesn't need to keep all tensors in memory simultaneously. Once Conv1's output has been consumed by Pool1, Conv1's buffer can be reused for Conv2's input. Once Pool1's output has been consumed by Conv2, Pool1's buffer can be reused for Pool2's input. This is called buffer reuse or in-place computation.
With buffer reuse, peak memory usage is determined by the largest tensor that must exist at any point during inference. For this network, Conv1's 1.15 MB output is the peak. The engine allocates 1.15 MB once and reuses it for all subsequent layers (which produce smaller outputs). Peak activation memory drops from 2.2 MB to 1.15 MB—a 48% reduction.
But 1.15 MB is still too large for most embedded targets. The solution is to reduce layer output sizes. If the input resolution drops from 96×96 to 48×48, Conv1's output becomes 48×48×32 = 73,728 elements = 288 KB as float32. If the model is quantized to int8, that drops to 72 KB. Suddenly, the model fits in the SRAM budget.
The activation memory requirement depends on:
The largest intermediate tensor in the network (determined by layer output dimensions and channel count)
The number of tensors that must coexist (determined by the network's dataflow—feedforward networks reuse buffers better than networks with skip connections or branches)
Scratch buffers required by specific operations (e.g., im2col transformations for convolution can temporarily double memory usage)
Most inference frameworks include a memory profiler that reports activation memory. TensorFlow Lite's Benchmark Tool outputs "tensor arena size." TensorFlow Lite Micro's interpreter initialization fails if the tensor arena is too small, and it reports the required size in the error message. Use these tools to measure activation memory before deployment.
Memory Layout Strategies: In-Place Computation, Buffer Reuse, Scratch Pads
Reducing activation memory requires understanding how the inference engine allocates and reuses buffers. There are three primary strategies: in-place computation, buffer reuse, and scratch pads.
In-place computation means that a layer writes its output to the same buffer that held its input. This is possible for operations where the output has the same dimensions as the input—element-wise operations like ReLU, batch normalization, or addition of residual connections.
For example, ReLU applies element-wise: output[i] = max(0, input[i]). The engine reads input[i], computes the result, and writes it back to input[i], overwriting the original value. No additional buffer is needed. Activation memory for ReLU is zero.
In-place computation fails when the operation changes the tensor's dimensions. A convolutional layer that reduces spatial resolution (stride > 1) or increases channel count produces a different-sized output, so it cannot overwrite the input buffer.
Buffer reuse means the engine recycles buffers across layers. After a layer consumes its input, the engine marks that input buffer as free and allocates it to a later layer's output. This is automatic in most inference engines—you allocate a large tensor arena, and the engine manages it internally.
The efficiency of buffer reuse depends on the network's structure. Feedforward networks (sequential layers with no branches) reuse buffers maximally because each layer's output is only used once. Networks with skip connections or concatenation operations (e.g., ResNet, DenseNet) cannot reuse buffers as aggressively because multiple layers' outputs must coexist.
For example, a ResNet block computes output = conv(input) + input, which means the input tensor must remain in memory while the convolution executes, then it's added to the convolution's output. The engine cannot reuse the input buffer until after the addition, which increases peak memory.
Scratch pads are temporary buffers used by specific operations that need intermediate storage. The most common example is the im2col transformation for convolution.
Convolution with a 3×3 kernel requires reading 9 input pixels for each output pixel. Naive implementations re-read those pixels from memory on every MAC operation. The im2col (image-to-column) transformation rearranges the input into a temporary matrix where each row contains the 9 pixels for one output position. The convolution then becomes a matrix multiply, which is faster on many processors.
But im2col requires a scratch buffer as large as (output_height × output_width × kernel_height × kernel_width × input_channels). For a 96×96 input with 32 channels and a 3×3 kernel, the scratch buffer is 96 × 96 × 3 × 3 × 32 = 2.65 million elements = 10.6 MB as float32. That's impractical on embedded hardware.
Embedded-optimized inference engines either avoid im2col entirely (using direct convolution with careful register blocking) or use partial im2col that processes the image in tiles small enough to fit in cache. The scratch buffer size becomes (tile_size × kernel_size × channels), which is orders of magnitude smaller.
You control memory layout by configuring the inference engine at compile time. TensorFlow Lite Micro requires you to allocate a static tensor arena and specify its size:
constexpr int kTensorArenaSize = 200 * 1024;  // 200 KB
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);

If the arena is too small, initialization fails with an error message indicating the required size. You increase the arena size, recompile, and re-test. If the arena fits but the system later crashes, you've exceeded total SRAM capacity—the arena plus the rest of the firmware's heap and stack exceed available RAM.
The workflow is:
Start with the smallest arena size that initialization accepts (reported by the framework).
Add 20% margin for runtime variation.
Verify that total SRAM usage (arena + stack + heap + firmware globals) is below 80% of capacity.
If total usage exceeds capacity, reduce the model's activation footprint or add external memory.
Static Memory Allocation for Inference: Why malloc() Is Not Your Friend
Dynamic memory allocation during inference is a design error on embedded systems. The heap is unreliable, fragmentation is unavoidable, and allocation failures are unrecoverable. The correct pattern is static pre-allocation of all memory required for inference.
This means:
The tensor arena is a static array allocated at compile time.
Model weights are either stored in flash and accessed read-only, or copied to a static array in SRAM during initialization.
Input and output buffers are static arrays.
Scratch buffers, if needed, are static arrays.
Nothing is allocated with malloc() or new during inference. All memory is accounted for at compile time and allocated before the first inference runs.
The advantage is predictability. You know at compile time whether the model fits. You know that inference will never fail due to heap exhaustion. You know that memory usage is constant across inference passes, which simplifies power profiling and worst-case execution time analysis.
The disadvantage is inflexibility. If you want to run a different model, you must recompile the firmware with a different arena size. If you want to support multiple models simultaneously, you must allocate separate arenas for each, even if they never run at the same time.
For most embedded AI applications, predictability outweighs flexibility. You deploy one model or a small set of models, and they run on fixed hardware for months or years. Static allocation is the right choice.
Some frameworks, like TensorFlow Lite Micro, enforce static allocation by default—there is no dynamic allocation code in the inference engine. Others, like PyTorch Mobile or ONNX Runtime, assume dynamic allocation and must be configured explicitly to use pre-allocated buffers. Check your framework's documentation and verify that no malloc() calls occur during inference.
Memory Profiling Tools: How to Measure What a Deployed Model Actually Uses
Theoretical memory calculations tell you the minimum required. Profiling tells you what the model actually consumes on target hardware, including overhead from the inference engine, alignment padding, and runtime allocation.
The simplest profiling method is to fill RAM with a known pattern, run inference, and see how much of the pattern was overwritten.
// Fill unused RAM with 0xAA
extern uint8_t _end;  // End of .bss section (linker-provided symbol)
extern uint8_t _estack;  // Top of stack
memset(&_end, 0xAA, &_estack - &_end);

// Run inference
run_inference();

// Count how many bytes were overwritten
uint8_t* ptr = &_end;
size_t used = 0;
while (ptr < &_estack && *ptr != 0xAA) {
    ptr++;
    used++;
}

printf("RAM used: %zu bytes\n", used);

This gives a conservative estimate—it includes stack growth, heap allocations, and any other runtime memory usage, not just the tensor arena.
For finer granularity, use the framework's built-in profiler. TensorFlow Lite Micro's interpreter reports tensor arena usage after initialization:
size_t used = interpreter.arena_used_bytes();
printf("Tensor arena used: %zu bytes\n", used);

This tells you how much of the arena was actually allocated, which might be less than the total arena size if you over-provisioned.
For layer-wise memory profiling, some frameworks support memory tracing. You enable tracing, run inference, and the framework logs each allocation and deallocation. The output shows which layers allocate the largest tensors and when buffers are reused.
ARM's Arm NN framework includes a memory profiler that generates a timeline showing activation memory usage over the course of inference. Peaks in the timeline indicate which layers consume the most memory. If one convolutional layer allocates a 500 KB tensor while all others allocate under 100 KB, that layer is the bottleneck.
External tools like Percepio Tracealyzer or SEGGER SystemView visualize memory usage in real time on RTOS-based systems. They show heap allocations, stack usage, and task memory consumption as the system runs. If inference causes the heap to fragment or the stack to overflow, these tools catch it visually.
The goal of profiling is to answer three questions:
How much SRAM does inference consume at peak?
Which layer or operation causes the peak?
Is there margin for the rest of the system, or is SRAM at capacity?
If peak usage exceeds 80% of total SRAM, you're in the danger zone. Real systems have runtime variation—interrupt handlers push to the stack, communication buffers fill, sensor data arrives. If inference leaves no headroom, the system will crash under load even if it works in controlled testing.
A Worked Example: Diagnosing and Fixing an Activation Memory Failure
You are deploying a MobileNetV2-based image classifier to an STM32H743 (Cortex-M7 at 480 MHz, 1 MB SRAM, 2 MB flash). The model has 1.2 million parameters (int8 quantized, 1.2 MB in flash). The model summary reports:
Input: 96×96×3 (27,648 elements)
Largest layer output: 24×24×192 (110,592 elements)
Total parameters: 1.2M (1.2 MB as int8)
You calculate activation memory conservatively: 110,592 elements × 1 byte (int8) = 108 KB. You add 50% margin for scratch buffers: 162 KB. The STM32H743 has 1 MB of SRAM, so this seems safe.
You compile with a 200 KB tensor arena and deploy. Initialization succeeds. The first inference crashes with a memory fault.
Diagnosis Step 1: Check actual arena usage.
You enable TFLite Micro's diagnostic output. The interpreter reports:
Tensor arena used: 340 KB
Tensor arena size: 200 KB
ERROR: Insufficient arena size

The model requires 340 KB, not 162 KB. Your estimate was wrong.
Diagnosis Step 2: Identify the cause.
You examine the layer-wise memory profile. The bottleneck is not the 24×24×192 tensor (108 KB). It's a depthwise convolution that uses an im2col scratch buffer of 180 KB. Your conservative estimate didn't account for operation-specific scratch memory.
Fix Attempt 1: Increase the arena.
You increase the tensor arena to 350 KB. Initialization succeeds. Inference runs but crashes 80 ms in. The debugger shows a stack overflow.
Diagnosis Step 3: Check total SRAM usage.
The firmware uses:
Tensor arena: 350 KB
Application heap: 80 KB
RTOS kernel and task stacks: 120 KB
Communication buffers: 200 KB
Global variables: 50 KB Total: 800 KB
That leaves 224 KB of the 1 MB SRAM for the call stack. But the inference function is deeply recursive due to the framework's layer dispatching, and it overflows the stack during execution.
Fix Attempt 2: Reduce the model's activation footprint.
You cannot increase SRAM—the hardware is fixed. You must reduce memory consumption. You modify the model:
Reduce input resolution from 96×96 to 80×80.
Reduce the maximum channel count from 192 to 128.
Replace standard convolutions with depthwise separable convolutions (which have lower scratch buffer requirements).
You retrain the model with these changes. The new model has:
800,000 parameters (800 KB as int8)
Largest layer output: 20×20×128 (51,200 elements = 50 KB)
Tensor arena required: 180 KB (measured)
New total SRAM usage:
Tensor arena: 180 KB
Application heap: 80 KB
RTOS and stacks: 120 KB
Communication buffers: 200 KB
Globals: 50 KB Total: 630 KB
This leaves 394 KB of headroom, which is sufficient. You deploy and test. Inference runs successfully. Accuracy drops from 89.2% (original model) to 86.4% (reduced model), which is acceptable for the application.
This is the memory constraint workflow. The original model "fit" by parameter count (1.2 MB in 2 MB flash) but failed at runtime due to activation memory (340 KB required, only 200 KB allocated). Profiling revealed the actual requirement. Reducing the model's size brought activation memory within budget, at the cost of a 3% accuracy drop.
You could not have diagnosed this without measurement. The theoretical estimate was off by 2×. Only runtime profiling exposed the scratch buffer overhead.
What Happens When You Run Out of Memory
There are three ways a model fails the memory constraint: it doesn't fit in flash, it doesn't fit in SRAM, or it leaves insufficient margin for the rest of the system.
Doesn't fit in flash: The model's weights exceed available flash capacity. This is the easiest failure to diagnose and fix. Quantize the model from float32 to int8 (75% reduction in size), prune unnecessary layers, or select a smaller architecture. If quantization and pruning are insufficient, you've chosen the wrong model for the hardware. Select a smaller model or upgrade the hardware.
Doesn't fit in SRAM: The model's activations exceed available SRAM. This is harder to diagnose because activation memory is not visible in model summaries—you must profile it. Fix by reducing layer sizes, reducing channel counts, or using architectures with lower activation memory (e.g., MobileNet instead of ResNet). If the model is already minimal and still doesn't fit, add external PSRAM or upgrade to hardware with more SRAM.
Insufficient margin: The model fits, but total SRAM usage (model + firmware) exceeds 80–90% of capacity. This is the failure mode that shows up only under load. In testing, the system works. In production, when communication buffers fill, sensor data queues up, and interrupt handlers run frequently, the system crashes. Fix by reducing the model's memory footprint, reducing the application's memory usage (smaller buffers, fewer RTOS tasks), or upgrading to hardware with more SRAM.
The memory constraint is the first constraint you check and the easiest to violate. If the model doesn't fit in memory, no amount of compute optimization or power tuning will make it work. Memory is binary: it fits or it doesn't.
That's why this chapter comes first in Part II. Before you evaluate latency, power, or acceleration, you verify that the model fits in memory. If it doesn't, you fix it. Only then do you move to the next constraint.
Chapter 6 examines that next constraint: compute. The model fits in memory. Now, can it run fast enough?

---

*[← Chapter 4](./chapter-04-inference-mechanics.md) | [Table of Contents](../README.md) | [Chapter 6 →](./chapter-06-compute.md)*