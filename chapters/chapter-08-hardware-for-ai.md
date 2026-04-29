# Chapter 8: Hardware for AI: Accelerators, DSPs, and FPGAs

The model runs correctly on a Cortex-M4 at 100 MHz, but inference takes 850 milliseconds and your deadline is 100 milliseconds. You've already quantized to int8, pruned 30% of the weights, and optimized the layer sequence. The model cannot get smaller without accuracy falling below the application's threshold. The CPU cannot get faster without replacing the entire hardware platform.

Your colleague suggests adding a neural processing unit (NPU)—a specialized accelerator that executes matrix multiplications 10–50× faster than the CPU. You evaluate an STM32N6 with a built-in NPU. The same model now runs in 45 milliseconds. The latency constraint is satisfied, and the power consumption drops by 60% because the accelerator completes inference in one-fifth the time.

But the STM32N6 costs $12 per unit instead of $3, the development toolchain is more complex, and the NPU only accelerates certain operations—your model's depthwise convolutions fall back to the CPU, limiting the speedup. The decision is not obvious. Hardware acceleration solves the compute constraint but introduces cost, complexity, and new failure modes.

This chapter teaches you to evaluate hardware acceleration options, understand the tradeoffs between CPUs, DSPs, NPUs, and FPGAs, and determine when a hardware accelerator is worth the cost. By the end of this chapter, you will be able to select an appropriate acceleration strategy for a given constraint profile and explain when NOT to use an accelerator—when a well-optimized CPU implementation is the better choice.

## Why CPUs Are Inefficient for Neural Network Inference

A general-purpose CPU is designed to execute arbitrary instruction sequences with maximum flexibility. It has a complex pipeline, branch prediction, out-of-order execution, and caches to handle unpredictable workloads. This flexibility comes at a cost: energy and silicon area spent on control logic instead of arithmetic units.

Neural network inference is the opposite of arbitrary. It's a highly regular computation—millions of multiply-accumulate operations with predictable data flow and no branching. A CPU executing inference spends most of its energy fetching instructions, managing caches, and controlling the pipeline, not performing the actual arithmetic.

Consider a matrix multiplication: C = A × B. For a 128×128 matrix multiplied by a 128×128 matrix, the operation requires 128³ = 2,097,152 MACs. On a Cortex-M7 at 400 MHz, this takes roughly 10 milliseconds (200 MMAC/s sustained throughput).

Break down where the time goes:

Arithmetic operations (multiply-accumulate): ~30% of cycles

Memory access (loading A, B, writing C): ~50% of cycles

Instruction fetch and decode: ~15% of cycles

Pipeline stalls and cache misses: ~5% of cycles

Only 30% of the CPU's time is spent doing useful arithmetic. The rest is overhead. A specialized accelerator eliminates or minimizes that overhead.

An NPU designed for matrix multiplication has:

Dedicated MAC units arranged in a 2D array (e.g., 16×16 MACs executing in parallel)

Local scratchpad memory close to the compute units (no cache, no arbitrary access)

A simple controller that sequences operations without instruction fetch overhead

The same 128×128 matrix multiply on a 16×16 MAC array takes:

(128 / 16)³ = 512 cycles for arithmetic

Plus memory transfer time if A and B are in main memory

At 400 MHz, this is ~1.3 milliseconds—nearly 8× faster than the CPU. The speedup comes from parallel execution (256 MACs per cycle instead of 1) and eliminating control overhead.

But the NPU is not a general-purpose processor. It can only execute operations that map to its MAC array and memory architecture. If your model includes operations the NPU doesn't support—certain activation functions, dynamic control flow, or sparse operations—those operations fall back to the CPU, and the speedup is lost.

This is the fundamental tradeoff: CPUs are flexible but inefficient. Accelerators are efficient but inflexible. The right choice depends on whether your model's operations map cleanly to the accelerator's capabilities.

## DSPs for AI: Fixed-Point Arithmetic, SIMD, and Signal Processing Pipelines

Digital signal processors (DSPs) are specialized CPUs optimized for signal processing workloads—filtering, FFT, correlation. They're not AI-specific accelerators, but they excel at certain AI operations because of three features: hardware multiply-accumulate units, SIMD extensions for packed arithmetic, and fixed-point arithmetic support.

A DSP like the TI C674x or ARM Cortex-M55 includes:

Multiple MAC units executing in parallel (2–8 MACs per cycle)

SIMD instructions that operate on 4 or 8 packed 8-bit or 16-bit values

Dedicated address generation units for efficient memory access patterns (stride, circular buffers)

Zero-overhead loop counters

For neural network inference, DSPs are effective for:

Fully connected layers (matrix-vector multiply is a DSP primitive)

1D convolution (used for audio processing, time-series)

Depthwise separable convolution (channel-wise operations map well to SIMD)

Fixed-point quantized models (DSPs handle int8 and int16 natively)

DSPs struggle with:

2D convolution (requires complex address patterns that exceed the DSP's address generation capabilities)

Pooling and activation functions (not traditional DSP operations)

Dynamic models with conditional execution

A worked comparison: keyword spotting on audio using a 1D convolutional network.

On Cortex-M4 (general-purpose CPU):

Clock: 100 MHz

Throughput: 60 MMAC/s for int8 with CMSIS-NN optimizations

Latency: 18M MACs / 60M = 300 ms

On TI C5535 DSP:

Clock: 100 MHz

Throughput: 200 MMAC/s for int16 (dual MAC units, SIMD)

Latency: 18M MACs / 200M = 90 ms

The DSP is 3.3× faster for the same clock speed because it has more MAC units and better SIMD utilization. But the DSP costs more, consumes more power (120 mA vs. 60 mA active current), and requires different development tools (TI Code Composer Studio instead of standard GCC).

DSPs are a middle ground between general-purpose CPUs and dedicated AI accelerators. They're more efficient than CPUs for signal-heavy AI tasks but less efficient than NPUs for general neural network inference. They shine when your application includes both AI inference and traditional signal processing (FFT for audio features, filtering for sensor data)—you can run both workloads on the same processor instead of needing two separate chips.

## NPUs and MicroNPUs: What They Are, What They Accelerate Well, and What They Don't

A neural processing unit (NPU) is a domain-specific accelerator designed specifically for neural network inference. It includes:

A MAC array (e.g., 8×8, 16×16, or 32×32 MACs) that executes matrix operations in parallel

On-chip SRAM or scratchpad memory for weights and activations

A controller that sequences layer execution without CPU intervention

NPUs are categorized by size and target application:

Large NPUs (Google TPU, NVIDIA TensorRT): Datacenter accelerators with thousands of MACs, gigabytes of on-chip memory, designed for batch inference on servers. Not relevant for embedded systems.

Edge NPUs (Coral Edge TPU, Intel Movidius, Hailo-8): Embedded accelerators with 100–1000 MACs, megabytes of memory, designed for edge devices like cameras and drones. Power consumption: 0.5–5W.

MicroNPUs (ARM Ethos-U55, STM32 AI core, Syntiant NDP120): Microcontroller-scale accelerators with 8–256 MACs, kilobytes of memory, designed to augment MCUs. Power consumption: 10–200 mW.

For this book, we focus on microNPUs—accelerators integrated into or co-packaged with microcontrollers.

The ARM Ethos-U55 is representative. It's a microNPU that pairs with Cortex-M55 or Cortex-M85 CPUs. Key specs:

32, 64, 128, or 256 MACs (configurable at design time)

Supports int8 and int16 quantized models

128 KB to 2 MB on-chip SRAM

Claimed performance: 0.5 TOPS (trillion operations per second) at 500 MHz for int8

For a model with 50 million int8 MACs, theoretical latency on a 256-MAC Ethos-U55 at 500 MHz:

Peak throughput: 256 MACs × 500 MHz = 128 GMAC/s = 0.128 TOPS

Latency: 50M / 128M = 390 ms

But that's theoretical. Real performance depends on whether the model's operations map efficiently to the accelerator. The Ethos-U55 accelerates:

2D convolution (standard and depthwise)

Fully connected layers

Pooling (max, average)

Element-wise operations (add, multiply, ReLU)

It does not accelerate (operations fall back to the CPU):

Attention mechanisms

Custom activation functions

Sparse operations

Dynamic control flow

If your model is 80% standard convolutions (accelerated) and 20% custom operations (CPU), the effective speedup is not 10× but more like 3–5×, because the CPU becomes the bottleneck for the unaccelerated portions.

Another example: STM32N6 with integrated AI accelerator.

Cortex-M55 CPU at 800 MHz

Integrated NPU: claimed 3 TOPS for int8

2.5 MB SRAM, 4 MB flash

Supports TensorFlow Lite models converted with STM32Cube.AI

For a 40-million-MAC MobileNetV2 model:

CPU-only (Cortex-M55 optimized): ~180 ms

NPU-accelerated: ~45 ms (measured, not theoretical)

The 4× speedup is real and transformative for applications with hard latency requirements. But the STM32N6 costs $12–15 per unit versus $4–6 for an STM32H7 without NPU. For a product manufacturing 100,000 units, that's $800,000 in additional BOM cost.

The NPU decision is economic: does the latency improvement justify the cost increase? For consumer products with aggressive latency targets (smart doorbells, wearables), yes. For industrial sensors with relaxed latency budgets, probably not.

## FPGAs for Embedded AI: Reconfigurability vs. Power vs. Development Cost

Field-programmable gate arrays (FPGAs) are reconfigurable hardware. You design a custom circuit—MAC arrays, memory controllers, custom operations—and program it onto the FPGA. Unlike CPUs or NPUs, which have fixed architectures, FPGAs let you implement exactly the accelerator you need.

For AI inference, FPGAs offer:

Custom MAC array sizes (you choose how many MACs based on your model)

Custom data paths optimized for specific operations

Bit-level precision (not limited to 8-bit or 16-bit—you can use 6-bit, 10-bit, or mixed precision)

Pipelined execution with no instruction fetch overhead

FPGAs are used in embedded AI when:

The model has unusual operations that CPUs and NPUs don't support

Latency requirements are so aggressive that even NPUs are insufficient

The application requires field-reconfigurability (update the accelerator logic after deployment)

The downside is cost and complexity. FPGAs are expensive ($20–$200 per chip for embedded-grade parts), power-hungry (500 mW to 5W for inference), and difficult to program. Designing an FPGA accelerator requires HDL (Verilog or VHDL) expertise, which most embedded software engineers lack. Development time is measured in months, not weeks.

A worked comparison: deploying a YOLO object detection model.

Option A: Cortex-A53 (CPU-only)

Raspberry Pi Zero 2 W (quad-core at 1 GHz)

Latency: 2,500 ms per frame

Power: 1.5 W

Cost: $15

Option B: Coral Edge TPU (dedicated NPU)

Google Edge TPU with Cortex-A53 host

Latency: 25 ms per frame

Power: 2 W (TPU + CPU)

Cost: $60

Option C: Xilinx Zynq (FPGA + ARM CPU)

Zynq-7020 with custom YOLO accelerator in FPGA fabric

Latency: 15 ms per frame

Power: 3 W

Cost: $150

Development time: 6 months (FPGA design + verification)

Option B (Coral Edge TPU) offers the best balance: 100× speedup over CPU-only, reasonable power, and off-the-shelf availability. Option C (FPGA) is 1.7× faster than the TPU but costs 2.5× more, consumes 50% more power, and requires six months of FPGA development. Unless the application absolutely requires sub-20ms latency, the FPGA is not worth it.

FPGAs make sense for:

Research prototypes where flexibility is more valuable than cost

Niche applications where volume is too low to justify ASIC development but performance requirements exceed what NPUs provide

Applications with evolving models where field-reconfigurability justifies the upfront cost

For most embedded AI deployments, FPGAs are overkill. CPUs with optimized software, DSPs, or microNPUs provide better cost-performance-power tradeoffs.

## Integrated AI Hardware: Real Devices, Real Tradeoffs

Several microcontroller and SoC vendors now offer chips with integrated AI acceleration. These devices combine a CPU, NPU, memory, and peripherals in a single package. Here are four representative examples with real specs and tradeoffs.

## ARM Cortex-M55 + Ethos-U55

What it is: ARM's microNPU architecture paired with the Cortex-M55 CPU. Available from vendors like NXP, ST, Renesas.

Specs:

CPU: Cortex-M55 at 200–800 MHz with Helium (M-profile vector extension)

NPU: Ethos-U55 with 32–256 MACs

Memory: 256 KB – 2 MB SRAM (vendor-dependent)

Power: 50–150 mW during inference (vendor-dependent)

Best for: Keyword spotting, gesture recognition, low-resolution image classification (96×96 or smaller).

Limitations: Small MAC array limits throughput. Models larger than ~500K parameters may not fit in on-chip memory. No support for dynamic models or attention mechanisms.

Toolchain: ARM Compiler 6, TensorFlow Lite Micro with Ethos-U delegate, Vela model optimizer (quantization + graph transformation).

Cost: $8–$15 per unit.

STM32N6

What it is: STMicroelectronics' AI-focused MCU with integrated NPU.

Specs:

CPU: Cortex-M55 at 800 MHz

NPU: STM32 AI accelerator, claimed 3 TOPS int8

Memory: 2.5 MB SRAM, 4 MB flash

Power: 180 mW during inference (measured)

Best for: Vision applications on microcontrollers—defect detection, gesture recognition, small object detection.

Limitations: NPU is proprietary (not Ethos-U), so toolchain compatibility is limited to STM32Cube.AI. Models must be converted and validated specifically for this chip.

Toolchain: STM32Cube.AI (TensorFlow, PyTorch, ONNX support), STM32CubeIDE.

Cost: $12–$18 per unit.

Syntiant NDP120

What it is: Ultra-low-power neural decision processor designed for always-on audio AI.

Specs:

NPU: 8-core neural processing array

Power: <200 µW for always-on keyword detection

Memory: 1.5 MB on-chip

No general-purpose CPU (inference-only device)

Best for: Voice interfaces, acoustic event detection. Runs continuously on a coin cell for years.

Limitations: Audio-only. Cannot run image models. No general-purpose CPU means it must be paired with a host MCU for application logic.

Toolchain: Syntiant Core 2 (proprietary tool for model training and deployment).

Cost: $3–$5 per unit.

Kendryte K210

What it is: RISC-V-based AI SoC with dual-core CPU and KPU (Kendryte Processing Unit).

Specs:

CPU: Dual-core RISC-V at 400 MHz

KPU: Convolutional neural network accelerator (64 MACs)

Memory: 8 MB SRAM

Power: 300 mW during inference

Best for: Low-cost vision applications. Popular in hobbyist and educational markets.

Limitations: KPU only accelerates convolution—fully connected layers and non-standard operations run on the slow RISC-V CPU. Limited commercial-grade support.

Toolchain: Kendryte Standalone SDK, limited TensorFlow support.

Cost: $6–$8 per unit.

Comparison Summary

Device

NPU Type

Best Use Case

Power

Cost

Toolchain Maturity

M55 + Ethos-U55

MicroNPU

Audio, sensors

50–150 mW

$8–15

High

STM32N6

MicroNPU

Vision on MCU

180 mW

$12–18

Medium

Syntiant NDP120

Audio NPU

Always-on voice

0.2 mW

$3–5

Low (proprietary)

Kendryte K210

Vision NPU

Hobbyist vision

300 mW

$6–8

Low

The decision matrix:

If you need always-on voice at ultra-low power → Syntiant NDP120

If you need general-purpose inference on MCU with good tooling → Cortex-M55 + Ethos-U55

If you need vision on MCU with higher performance → STM32N6

If cost and hobbyist support matter more than commercial reliability → Kendryte K210

None of these is a universal solution. Each is optimized for a specific use case, and deploying on the wrong one leads to performance or cost failures.

When NOT to Use a Hardware Accelerator

Hardware accelerators are not always the right answer. There are cases where a well-optimized CPU implementation is faster, cheaper, or more flexible. Knowing when to reject acceleration is as important as knowing when to apply it.

Case 1: The model is too small to benefit.

If your model has 10,000 parameters and takes 5 ms on a Cortex-M4, adding an NPU won't help. The overhead of transferring data to the accelerator, configuring it, and transferring results back exceeds the time saved by parallel execution. Small models are already fast enough on CPUs.

Case 2: The model's operations don't map to the accelerator.

If your model uses custom layers, dynamic control flow, or sparse operations, the accelerator may only accelerate 20–30% of the workload. The rest falls back to the CPU, and the speedup is negligible. Profile the model to see which operations dominate. If most of the time is spent in unaccelerated operations, the accelerator is wasted.

Case 3: The CPU is fast enough.

If a Cortex-M7 at 480 MHz meets your latency requirement with optimized CMSIS-NN kernels, why add an NPU? The CPU solution is simpler (one chip instead of two), cheaper, and uses familiar tools. Accelerators solve problems. If you don't have a compute problem, don't add an accelerator.

Case 4: The accelerator's power consumption exceeds the CPU's.

Some accelerators consume more power than an optimized CPU implementation for small models. An NPU running at 500 MHz might draw 150 mW while a CPU running at 100 MHz draws 30 mW. If the NPU finishes in 10 ms and the CPU finishes in 40 ms, energy per inference is:

NPU: 150 mW × 0.01 s = 1.5 mJ

CPU: 30 mW × 0.04 s = 1.2 mJ

The CPU uses less energy despite being slower. For battery-powered systems with low duty cycles, the slower CPU is the better choice.

Case 5: Toolchain immaturity.

If the accelerator requires a proprietary toolchain with poor documentation, limited model support, and buggy code generation, the development cost exceeds the performance benefit. Debugging accelerator-specific failures is harder than debugging CPU code. Unless the performance gain is transformative (10× or more), stick with the CPU and standard tools.

The workflow for deciding whether to use an accelerator:

Profile the model on the CPU with optimized kernels (CMSIS-NN, XNNPACK).

Measure whether the CPU meets latency, power, and cost requirements.

If yes, stop. Don't add complexity.

If no, identify the bottleneck operation (convolution, matrix multiply, etc.).

Check whether an available accelerator supports that operation.

Estimate the speedup (not theoretical—realistic, accounting for fallback operations).

Compare cost, power, and development effort.

If the accelerator's benefits justify its costs, use it. Otherwise, optimize the model instead.

Acceleration is a tool, not a goal. The goal is to meet the application's constraints with the simplest, lowest-cost solution. Sometimes that's a CPU. Sometimes that's an NPU. Sometimes it's rejecting AI entirely and using a classical algorithm.

## A Worked Example: Same Model, Three Hardware Targets

You have a 96×96 image classification model for defect detection on a manufacturing line. The model has 400,000 parameters (int8 quantized, 400 KB), 35 million MACs, and 95% accuracy. Application requirements:

Latency: ≤50 ms (line speed requires classification before the part moves out of frame)

Power: <500 mW average (system is mains-powered but heat-limited)

Cost: <$15 per unit (high-volume production, 50,000 units/year)

You evaluate three hardware options.

Option A: STM32H7 (CPU-only)

Specs:

Cortex-M7 at 480 MHz

1 MB SRAM, 2 MB flash

Optimized with CMSIS-NN

Measured Performance:

Latency: 180 ms

Power during inference: 400 mW

Cost: $6 per unit

Analysis: Fails the 50 ms latency requirement by 3.6×. Even with perfect optimization, the CPU cannot run this model fast enough.

Option B: STM32N6 (CPU + NPU)

Specs:

Cortex-M55 at 800 MHz + STM32 AI accelerator (3 TOPS)

2.5 MB SRAM, 4 MB flash

Model converted with STM32Cube.AI

Measured Performance:

Latency: 42 ms (NPU-accelerated)

Power during inference: 650 mW

Cost: $14 per unit

Analysis: Meets the 50 ms latency requirement. Exceeds the 500 mW power budget by 30%, but since the system is mains-powered and only heat-limited, 650 mW might be acceptable if thermal analysis confirms it. Cost is within budget. Viable option.

Option C: Raspberry Pi Compute Module 4 (Application Processor)

Specs:

Quad-core Cortex-A72 at 1.5 GHz

1–4 GB RAM (overkill for this application)

TensorFlow Lite with ARM NEON optimizations

Measured Performance:

Latency: 8 ms (NEON-optimized)

Power during inference: 2,500 mW

Cost: $35 per unit

Analysis: Easily meets the latency requirement, but power consumption is 5× over budget and cost is 2.3× over budget. The Raspberry Pi is overkill for this application. Not viable.

Conclusion:

Option B (STM32N6 with NPU) is the only viable solution. Option A (CPU-only) fails latency. Option C (application processor) fails power and cost. The NPU's 4× speedup over the CPU makes the difference between a product that works and one that doesn't.

But suppose the application requirement was 100 ms latency instead of 50 ms. Then Option A becomes viable—180 ms is close enough that further optimization (reducing the model to 30 million MACs, using better quantization) could bring it under 100 ms. In that case, you'd choose the $6 CPU-only solution instead of the $14 NPU solution, saving $8 per unit × 50,000 units = $400,000 in BOM cost.

This is the hardware selection decision. Constraints determine which options are viable. Cost determines which viable option you choose. Acceleration is justified only when it's the cheapest way to meet the constraints.

## What Comes Next

Hardware acceleration solves local compute and power constraints. But embedded systems rarely operate in isolation. They communicate—with other devices, with edge gateways, with cloud servers. The next chapter examines the edge-cloud spectrum: when to process data locally, when to offload to the network, and how to design hybrid inference strategies that split computation across processing tiers.


---

*[<- Chapter 7](./chapter-07-power-and-energy.md) | [Table of Contents](../README.md) | [Chapter 9 ->](./chapter-09-communication-edge-cloud.md)*
