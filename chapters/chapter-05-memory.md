# Chapter 5 — Memory: Fitting Models into Constrained Spaces
*The model that fits by arithmetic doesn't fit by reality.*

The datasheet says 256 KB of RAM. Your model's weights occupy 180 KB. Simple arithmetic: 256 - 180 = 76 KB of headroom. The model fits.

You compile, flash, run. The system crashes 40 milliseconds into the first inference with a memory fault. The debugger shows the heap allocator failed, malloc returned null, and the inference engine tried to write to address 0x00000000.

The model's weights were never the problem. Weights live in flash. What crashed was activation memory—the intermediate tensors created during inference, which live in RAM. Your model's activation footprint is 210 KB. Add 30 KB for the Bluetooth stack, 8 KB for system heap, 4 KB for call stack, and you need 252 KB before the model even starts. The model that "fit" by parameter count doesn't fit at runtime.

This failure mode is common because activation memory is invisible until you measure it. Datasheets specify total RAM. Model summaries report parameter count and weight size. Neither tells you how much RAM inference will consume while executing.

This chapter teaches you to calculate activation memory before deployment, identify which components consume the most memory and why, and determine whether a model fits within a target's memory envelope with enough margin for the rest of the system.

<!-- → INFOGRAPHIC: Memory allocation breakdown showing total 256 KB SRAM divided into visible components (weights in flash, not SRAM) and invisible runtime components (210 KB activations, 30 KB Bluetooth, 8 KB heap, 4 KB stack) summing to 252 KB — exceeding capacity by the "invisible" activation memory -->

## Flash vs. SRAM: Where Weights Live, Where Activations Live

Memory in embedded systems is partitioned. The critical distinction for AI deployment is flash versus SRAM.

Flash is non-volatile. Data persists when power dies. Flash is abundant—microcontrollers commonly have 512 KB to 2 MB. But flash is read-only during normal operation. You can't write to it from running code without special erase-and-program cycles that take milliseconds and degrade the flash over time.

For AI deployment, flash stores the firmware binary, the model's weights and biases, and constant lookup tables.

SRAM is volatile. Data in SRAM is lost when power dies. SRAM is fast—reads and writes complete in 1–2 CPU cycles. SRAM is writable—you can modify it freely. But SRAM is scarce—microcontrollers commonly have 64 KB to 512 KB, and high-end parts top out at 1–2 MB.

For AI deployment, SRAM stores the application's heap and stack, global and static variables, RTOS control blocks if applicable, and the model's activation tensors during inference.

<!-- → TABLE: Flash vs. SRAM comparison showing property (volatility, access speed, writability, typical capacity on MCU, cost per byte), what each memory type stores for AI deployment, and constraints — Flash (non-volatile, slower, read-only, 512KB-2MB, stores weights) vs SRAM (volatile, 1-2 cycle access, writable, 64KB-512KB, stores activations) -->

The memory constraint is not "does the model fit in total memory?" but "does each component fit in the right type of memory?"

Consider a model with 500,000 parameters (2 MB as float32, 500 KB as int8) deployed on an STM32L4 with 640 KB of SRAM and 2 MB of flash. The int8 weights fit in flash with room to spare. But if the model's activations require 400 KB of SRAM, and the application already uses 200 KB for Bluetooth stack and buffers, total SRAM usage is 600 KB. Within the 640 KB limit, but with only 40 KB of margin. If the system also needs a 64 KB DMA buffer for sensor data, SRAM usage exceeds capacity, and deployment fails.

Some microcontrollers support execute-in-place (XIP) from flash, allowing the processor to fetch instructions and read-only data directly from flash without copying to SRAM. This is efficient for weights—the inference engine reads weights from flash as needed during each layer's computation. But activations cannot live in flash because they change during inference. Activations must be in writable memory, which means SRAM or external DRAM.

External memory—PSRAM, SDRAM, eMMC—expands total capacity but at a cost. External memory is slower than on-chip SRAM. Access latency increases from 1–2 cycles to 10–50 cycles depending on interface. External memory consumes more power because data traverses the external bus. And external memory increases BOM cost and board complexity.

If your model's activations exceed on-chip SRAM, you have four options: reduce the model's activation footprint, add external PSRAM and accept the latency and power penalty, stream activations to flash between layers (extremely slow, only viable for ultra-low-latency-tolerant applications), or reject the target and select hardware with more SRAM.

## Weight Memory: The Easy Part

The size of a model's weights is straightforward to calculate. Count the parameters, multiply by bytes per parameter, and you have the flash requirement.

For a fully connected layer with input_size inputs and output_size outputs:

Parameters = (input_size × output_size) + output_size

The first term is the weight matrix. The second term is the bias vector.

For a 2D convolutional layer:

Parameters = (kernel_height × kernel_width × input_channels × output_channels) + output_channels

Most ML frameworks report total parameter count when you compile the model. If you don't have access to the framework, you can calculate manually from the architecture specification.

Precision determines bytes per parameter. float32: 4 bytes. int8: 1 byte. A model with 1 million parameters occupies 4 MB as float32, 1 MB as int8.

Quantization from float32 to int8 reduces weight memory by 75%. For a device with 1 MB of flash and 200 KB already consumed by firmware, a 1.2 MB float32 model doesn't fit, but the 300 KB int8 version does.

Weight memory is fixed after training. During inference, weights are read-only. They never change. This makes weight memory the easiest constraint to satisfy—if weights don't fit in flash, you quantize or prune until they do.

The one complication is that some inference engines copy weights from flash to SRAM for faster access. This speeds up inference—SRAM access is faster than flash access—but consumes SRAM. If your SRAM budget is tight, configure the engine to access weights directly from flash and accept the latency penalty.

## Activation Memory: The Hidden Footprint

Activation memory is the RAM consumed by intermediate tensors during inference. When you run a forward pass, each layer produces an output tensor that becomes input to the next layer. Those tensors exist in memory while the layer executes. The total size of those tensors—plus any scratch buffers the inference engine uses internally—is the activation memory footprint.

Unlike weight memory, activation memory is not obvious from the model architecture. You cannot calculate it by counting parameters. You must trace the dataflow through the network and account for every intermediate tensor that exists simultaneously.

<!-- → CHART: Stacked bar chart showing activation memory without buffer reuse (2,268 KB total, sum of all layer outputs) versus with buffer reuse (1.15 MB peak, size of largest tensor Conv1), demonstrating 48% reduction -->

Consider a simple convolutional network:

- Input: 96×96×3 image (27,648 elements, 110 KB as float32)
- Conv1: 96×96×32 output (294,912 elements, 1.15 MB as float32)
- Pool1: 48×48×32 output (73,728 elements, 288 KB as float32)
- Conv2: 48×48×64 output (147,456 elements, 576 KB as float32)
- Pool2: 24×24×64 output (36,864 elements, 144 KB as float32)
- Flatten and FC: 10 outputs (10 elements, 40 bytes as float32)

If the inference engine allocates separate buffers for each layer's output, peak memory usage is the sum of all activation tensors: 2,268 KB. That's 2.2 MB of SRAM, which exceeds the capacity of most microcontrollers.

But the engine doesn't need to keep all tensors in memory simultaneously. Once Conv1's output has been consumed by Pool1, Conv1's buffer can be reused for Conv2's input. This is called buffer reuse.

With buffer reuse, peak memory usage is determined by the largest tensor that must exist at any point during inference. For this network, Conv1's 1.15 MB output is the peak. The engine allocates 1.15 MB once and reuses it for subsequent layers. Peak activation memory drops from 2.2 MB to 1.15 MB—a 48% reduction.

But 1.15 MB is still too large for most embedded targets. The solution is to reduce layer output sizes. If input resolution drops from 96×96 to 48×48, Conv1's output becomes 48×48×32 = 73,728 elements = 288 KB as float32. Quantized to int8, that's 72 KB. Suddenly, the model fits.

The activation memory requirement depends on the largest intermediate tensor, the number of tensors that must coexist (determined by dataflow—feedforward networks reuse buffers better than networks with skip connections), and scratch buffers required by specific operations.

Most inference frameworks include a memory profiler. TensorFlow Lite's Benchmark Tool outputs "tensor arena size." TensorFlow Lite Micro's interpreter initialization fails if the tensor arena is too small and reports the required size in the error message. Use these tools to measure activation memory before deployment.

## Buffer Reuse and In-Place Computation

Reducing activation memory requires understanding how the inference engine allocates and reuses buffers. There are two primary strategies: in-place computation and buffer reuse.

In-place computation means a layer writes its output to the same buffer that held its input. This is possible for operations where output has the same dimensions as input—element-wise operations like ReLU, batch normalization, or addition of residual connections.

ReLU applies element-wise: output[i] = max(0, input[i]). The engine reads input[i], computes the result, writes it back to input[i], overwriting the original. No additional buffer needed. Activation memory for ReLU is zero.

In-place computation fails when the operation changes tensor dimensions. A convolutional layer that reduces spatial resolution (stride > 1) or increases channel count produces different-sized output, so it cannot overwrite the input buffer.

<!-- → INFOGRAPHIC: Buffer reuse strategy illustrated with timeline showing Layer 1 allocating Buffer A (large), Layer 2 consuming Buffer A and reusing it for its smaller output, Layer 3 reusing the same buffer again, versus naive allocation showing Buffer A, B, C all allocated simultaneously -->

Buffer reuse means the engine recycles buffers across layers. After a layer consumes its input, the engine marks that input buffer as free and allocates it to a later layer's output. This is automatic in most inference engines—you allocate a large tensor arena, and the engine manages it internally.

The efficiency of buffer reuse depends on network structure. Feedforward networks (sequential layers with no branches) reuse buffers maximally because each layer's output is only used once. Networks with skip connections or concatenation operations cannot reuse buffers as aggressively because multiple layers' outputs must coexist.

For example, a ResNet block computes output = conv(input) + input, which means the input tensor must remain in memory while the convolution executes, then it's added to the convolution's output. The engine cannot reuse the input buffer until after the addition, which increases peak memory.

## Static Allocation: Why malloc() Is a Mistake

Dynamic memory allocation during inference is a design error on embedded systems. The heap is unreliable, fragmentation is unavoidable, allocation failures are unrecoverable. The correct pattern is static pre-allocation of all memory required for inference.

This means the tensor arena is a static array allocated at compile time. Model weights are either stored in flash and accessed read-only, or copied to a static array in SRAM during initialization. Input and output buffers are static arrays. Nothing is allocated with malloc() during inference. All memory is accounted for at compile time.

The advantage is predictability. You know at compile time whether the model fits. You know inference will never fail due to heap exhaustion. Memory usage is constant across inference passes, which simplifies power profiling and worst-case execution time analysis.

The disadvantage is inflexibility. If you want to run a different model, you must recompile firmware with a different arena size. For most embedded AI applications, predictability outweighs flexibility. You deploy one model or a small set of models, and they run on fixed hardware for months or years.

Some frameworks, like TensorFlow Lite Micro, enforce static allocation by default—there is no dynamic allocation code in the inference engine. Others assume dynamic allocation and must be configured explicitly to use pre-allocated buffers. Check your framework's documentation and verify that no malloc() calls occur during inference.

## A Worked Example: When the Model Fits Until It Doesn't

You're deploying a MobileNetV2-based image classifier to an STM32H743 (Cortex-M7 at 480 MHz, 1 MB SRAM, 2 MB flash). The model has 1.2 million parameters (int8 quantized, 1.2 MB in flash). The model summary reports largest layer output: 24×24×192 (110,592 elements).

You calculate activation memory conservatively: 110,592 elements × 1 byte (int8) = 108 KB. You add 50% margin for scratch buffers: 162 KB. The STM32H743 has 1 MB of SRAM, so this seems safe.

You compile with a 200 KB tensor arena and deploy. Initialization succeeds. First inference crashes with a memory fault.

<!-- → CHART: Diagnostic flow showing progression through four attempts — initial estimate (162 KB, crashes), measured requirement (340 KB, still crashes on stack overflow), total SRAM analysis (800 KB consumed leaving insufficient stack space), final solution (model reduction to 180 KB arena, 630 KB total SRAM, successful deployment) -->

**Diagnosis Step 1: Check actual arena usage.**

You enable TFLite Micro's diagnostic output:

```
Tensor arena used: 340 KB
Tensor arena size: 200 KB
ERROR: Insufficient arena size
```

The model requires 340 KB, not 162 KB. Your estimate was wrong.

**Diagnosis Step 2: Identify the cause.**

You examine the layer-wise memory profile. The bottleneck isn't the 24×24×192 tensor (108 KB). It's a depthwise convolution that uses an im2col scratch buffer of 180 KB. Your conservative estimate didn't account for operation-specific scratch memory.

**Fix Attempt 1: Increase the arena.**

You increase the tensor arena to 350 KB. Initialization succeeds. Inference runs but crashes 80 ms in. The debugger shows stack overflow.

**Diagnosis Step 3: Check total SRAM usage.**

The firmware uses:

- Tensor arena: 350 KB
- Application heap: 80 KB
- RTOS kernel and task stacks: 120 KB
- Communication buffers: 200 KB
- Global variables: 50 KB
- Total: 800 KB

That leaves 224 KB of the 1 MB SRAM for the call stack. But the inference function is deeply recursive due to the framework's layer dispatching, and it overflows the stack during execution.

<!-- → TABLE: SRAM budget breakdown showing component, original allocation, and revised allocation after model reduction — tensor arena (350 KB → 180 KB), application heap (80 KB unchanged), RTOS/stacks (120 KB unchanged), communication buffers (200 KB unchanged), globals (50 KB unchanged), total (800 KB → 630 KB), headroom (224 KB → 394 KB) -->

**Fix Attempt 2: Reduce the model's activation footprint.**

You cannot increase SRAM—the hardware is fixed. You must reduce memory consumption. You modify the model: reduce input resolution from 96×96 to 80×80, reduce maximum channel count from 192 to 128, replace standard convolutions with depthwise separable convolutions.

You retrain. The new model has 800,000 parameters (800 KB as int8), largest layer output: 20×20×128 (51,200 elements = 50 KB), tensor arena required: 180 KB (measured).

New total SRAM usage:

- Tensor arena: 180 KB
- Application heap: 80 KB
- RTOS and stacks: 120 KB
- Communication buffers: 200 KB
- Globals: 50 KB
- Total: 630 KB

This leaves 394 KB of headroom. You deploy and test. Inference runs successfully. Accuracy drops from 89.2% (original model) to 86.4% (reduced model), acceptable for the application.

This is the memory constraint workflow. The original model "fit" by parameter count (1.2 MB in 2 MB flash) but failed at runtime due to activation memory (340 KB required, only 200 KB allocated). Profiling revealed the actual requirement. Reducing the model's size brought activation memory within budget, at the cost of 3% accuracy.

You could not have diagnosed this without measurement. The theoretical estimate was off by 2×. Only runtime profiling exposed the scratch buffer overhead.

## What Happens When You Run Out

There are three ways a model fails the memory constraint: it doesn't fit in flash, it doesn't fit in SRAM, or it leaves insufficient margin for the rest of the system.

**Doesn't fit in flash:** The model's weights exceed available flash capacity. This is the easiest failure to diagnose and fix. Quantize from float32 to int8 (75% reduction), prune unnecessary layers, or select a smaller architecture. If quantization and pruning are insufficient, you've chosen the wrong model for the hardware.

**Doesn't fit in SRAM:** The model's activations exceed available SRAM. This is harder to diagnose because activation memory is not visible in model summaries—you must profile it. Fix by reducing layer sizes, reducing channel counts, or using architectures with lower activation memory. If the model is already minimal and still doesn't fit, add external PSRAM or upgrade to hardware with more SRAM.

**Insufficient margin:** The model fits, but total SRAM usage exceeds 80–90% of capacity. This is the failure mode that shows up only under load. In testing, the system works. In production, when communication buffers fill, sensor data queues up, and interrupt handlers run frequently, the system crashes. Fix by reducing the model's memory footprint, reducing the application's memory usage, or upgrading hardware.

<!-- → INFOGRAPHIC: Three failure modes illustrated — failure mode 1: weights exceeding flash (fix: quantize/prune), failure mode 2: activations exceeding SRAM (fix: reduce layer sizes, measure don't estimate), failure mode 3: insufficient margin (fix: 80% rule, works in testing but crashes under production load) -->

The memory constraint is the first constraint you check and the easiest to violate. If the model doesn't fit in memory, no amount of compute optimization or power tuning will make it work. Memory is binary: it fits or it doesn't.

That's why this chapter comes first in Part II. Before you evaluate latency, power, or acceleration, you verify that the model fits in memory. If it doesn't, you fix it. Only then do you move to the next constraint.

The model fits in memory. Now, can it run fast enough?
