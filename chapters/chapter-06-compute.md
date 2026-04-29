# Chapter 6: Compute: Latency, Throughput, and Real-Time Guarantees

The model fits in memory. Weights occupy 600 KB of flash, activations require 140 KB of SRAM, and total memory usage leaves 200 KB of headroom. The memory constraint is satisfied.
You run your first inference pass and time it: 1,850 milliseconds. Your application requirement is 50 milliseconds. The model that fits in memory cannot run fast enough to be useful.
This is the compute constraint. The model's operations must execute within the time budget the application allows. If inference takes too long, the result arrives after the decision window closes—the object has moved out of the camera's field of view, the anomaly has progressed to failure, or the user has given up waiting. Latency violations don't crash the system the way memory violations do. They make the system correct but useless.
This chapter teaches you to estimate inference latency from operation count and hardware throughput, identify which operations dominate compute time, determine whether a model-hardware-deadline combination is feasible, and apply the optimizations that reduce latency when the initial estimate fails. By the end of this chapter, you will be able to evaluate the compute constraint before deployment and specify precisely what must change if the model is too slow.
FLOPs, MACs, and What They Actually Mean for Inference Time
A neural network's compute requirement is quantified in two ways: FLOPs (floating-point operations) and MACs (multiply-accumulate operations). Both measure the number of arithmetic operations the model performs per inference, but they count differently.
A FLOP is one floating-point operation—an addition, subtraction, multiplication, or division. A fully connected layer performing y = W × x + b involves:
(input_size × output_size) multiplications
(input_size × output_size) additions for the matrix multiply
(output_size) additions for the bias
Total FLOPs = 2 × (input_size × output_size) + output_size.
A MAC is one multiply-accumulate operation: result += a × b. This is the fundamental operation in matrix multiplication and convolution. Processors with MAC units (most ARM Cortex-M and Cortex-A cores, DSPs, and NPUs) can execute a MAC in a single instruction. The FLOP count is roughly 2× the MAC count because each MAC includes both a multiply and an add.
For embedded AI, MAC count is the more useful metric because it maps directly to hardware capabilities. A processor that executes 100 million MACs per second (100 MMAC/s) will take roughly (total MACs / 100M) seconds to execute the model, ignoring memory access overhead.
Most ML frameworks report FLOPs. TensorFlow's model summary includes FLOP count. PyTorch's thop library computes FLOPs. To convert to MACs, divide by 2 for operations that are pure multiply-adds (convolution, matrix multiply). For mixed operations (activations, pooling), use the FLOP count directly—it's a conservative upper bound.
The relationship between operation count and latency is:
Latency (seconds) = (Total operations) / (Sustained throughput)
where sustained throughput accounts for memory access, cache efficiency, and pipeline utilization. Theoretical peak throughput (from the datasheet) is always higher than sustained throughput. The ratio between peak and sustained varies from 0.5× to 0.9× depending on the processor, the operation, and how well the code is optimized.
For example, a Cortex-M7 at 400 MHz with an FPU has a theoretical peak of 400 MMAC/s (one MAC per cycle). But real code achieves 250–350 MMAC/s sustained due to memory bottlenecks and instruction overhead. A model with 50 million MACs takes:
Theoretical: 50M / 400M = 125 milliseconds
Actual: 50M / 300M = 167 milliseconds
The 33% gap between theoretical and actual is why you must profile on target hardware. The theoretical estimate tells you whether deployment is plausible. Profiling tells you whether it actually works.
Compute-Bound vs. Memory-Bound Inference
An operation is compute-bound if its execution time is dominated by arithmetic operations. It's memory-bound if execution time is dominated by loading data from memory. The distinction matters because the fix is different.
A small matrix multiply (64×64 × 64×64) on a Cortex-M7 with 64 KB of L1 cache is compute-bound. Both matrices fit in cache, so every multiply-add executes at peak throughput. Doubling the clock speed halves the execution time.
A large matrix multiply (512×512 × 512×512) on the same Cortex-M7 is memory-bound. The matrices are 2 MB total, far exceeding the 64 KB cache. Each element is fetched from SRAM multiple times, and memory bandwidth limits throughput. Doubling the clock speed might reduce execution time by only 20% because the processor spends most of its time waiting for memory.
Convolution is almost always memory-bound on microcontrollers. A 3×3 convolution on a 96×96 image with 32 input channels and 64 output channels requires:
96 × 96 × 64 × 3 × 3 × 32 = 169 million MACs
96 × 96 × 3 × 3 × 32 = 2.66 million input element reads (each input is reused across output channels, so actual reads are lower with good locality)
Even with perfect arithmetic throughput, if each input read costs 10 cycles, memory access time dominates.
How do you determine whether an operation is compute-bound or memory-bound? Measure arithmetic intensity:
Arithmetic intensity = (Operations) / (Bytes transferred)
High arithmetic intensity (>10 ops/byte) suggests compute-bound. Low arithmetic intensity (<1 op/byte) suggests memory-bound.
For a 64×64 matrix multiply:
Operations: 64 × 64 × 64 = 262,144 MACs
Bytes transferred: (64 × 64 + 64 × 64) × 4 bytes = 32 KB (assuming float32 and both matrices are in cache, read once)
Arithmetic intensity: 262,144 / 32,768 = 8 ops/byte
This is borderline. With good cache behavior, it's compute-bound. With poor cache behavior (cache thrashing), it becomes memory-bound.
For a 3×3 convolution on a 96×96 image:
Operations: 169 million MACs
Bytes transferred (worst case, no reuse): 96 × 96 × 3 × 3 × 32 × 4 bytes = 10.6 MB
Arithmetic intensity: 169M / 10.6M = 16 ops/byte
This looks compute-bound, but in practice, convolutional kernels reread input data from memory because the working set exceeds cache size. Effective arithmetic intensity drops to 1–3 ops/byte, making it memory-bound.
The optimization strategy differs:
For compute-bound operations: Increase clock speed, use SIMD, offload to hardware accelerator.
For memory-bound operations: Reduce data movement (tiling, buffer reuse), increase memory bandwidth (on-chip SRAM instead of external DRAM), compress data (quantization reduces bytes transferred).
Profiling tells you which bottleneck you have. If increasing clock speed from 100 MHz to 200 MHz doubles performance, you're compute-bound. If it increases performance by only 10%, you're memory-bound, and you need to fix memory access patterns, not arithmetic throughput.
Clock Speed, Pipeline Efficiency, and SIMD
A processor's clock speed—measured in MHz or GHz—tells you how many cycles per second it executes. But cycles are not operations. The number of operations per cycle depends on the instruction set, pipeline depth, and whether SIMD (single instruction, multiple data) extensions are available.
A Cortex-M0+ executes one instruction per cycle in ideal conditions. It has no pipeline forwarding, no branch prediction, and no SIMD. An add instruction takes 1 cycle if operands are in registers. A load from memory takes 2 cycles. A multiply takes 32 cycles (software emulation if no hardware multiplier is present). At 48 MHz, peak throughput is 48 million simple instructions per second, but MAC throughput is far lower.
A Cortex-M4 with DSP extensions executes one instruction per cycle with a 3-stage pipeline. It includes a single-cycle hardware multiplier and an FPU (floating-point unit). SIMD instructions (SIMD32 in ARM documentation) allow four 8-bit or two 16-bit operations in parallel. At 100 MHz, peak throughput is 100 MMAC/s for single-precision floats, or 400 million 8-bit MACs per second with SIMD.
A Cortex-M7 has a 6-stage superscalar pipeline, dual-issue execution (two instructions per cycle if they don't conflict), and a double-precision FPU. Cache and branch prediction reduce pipeline stalls. At 400 MHz, theoretical peak is 800 MMAC/s for single-precision floats (dual-issue MAC instructions), though sustained throughput is 400–600 MMAC/s due to instruction dependencies and memory access.
A Cortex-A53 (used in Raspberry Pi Zero 2 W) has an 8-stage pipeline, out-of-order execution, and ARM NEON SIMD extensions. NEON can execute 16 8-bit MACs per cycle per core. With 4 cores at 1 GHz, theoretical peak is 64 billion 8-bit MACs per second (64 GMAC/s). Sustained throughput is 30–50 GMAC/s depending on memory bandwidth and cache efficiency.
SIMD is the key to high throughput on microcontrollers. A Cortex-M4 executing scalar int8 MACs achieves 100 million MACs per second. With SIMD (packing four int8 values into a 32-bit register), the same processor achieves 400 million int8 MACs per second—a 4× improvement with no clock speed increase.
But SIMD requires explicit code optimization. Most inference frameworks include SIMD-optimized kernels for common operations (convolution, matrix multiply, activation functions). TensorFlow Lite Micro uses CMSIS-NN, ARM's optimized neural network library, which exploits SIMD on Cortex-M cores. If your inference code isn't using SIMD-optimized kernels, you're leaving a 4–8× performance improvement on the table.
To verify SIMD usage, disassemble the compiled binary and check for SIMD instructions:
Cortex-M4: SMLAD, SMLALD, USAD8 (SIMD multiply-accumulate)
Cortex-A: VMLA.I8, VMLA.I16 (NEON multiply-accumulate)
If you see only scalar instructions (MUL, ADD), the code is not using SIMD, and you need to enable optimized kernels in your build configuration.
Worst-Case Execution Time (WCET) Analysis for AI Inference
Real-time systems care about worst-case execution time (WCET), not average execution time. A system with 50 ms average latency and 500 ms worst-case latency fails if the deadline is 100 ms—the worst case violates the requirement.
WCET is the maximum time an operation can take under any input, any cache state, and any scheduling condition. For traditional embedded code (PID controllers, sensor drivers, communication protocols), WCET is analyzable using static analysis tools that model the processor's pipeline, cache, and memory system and compute the longest possible execution path.
For neural network inference, WCET analysis is difficult because:
Path explosion: A network with 20 layers has billions of possible execution paths when considering all input-dependent branches (though modern feedforward networks have few data-dependent branches).
Variable memory latency: Cache hit/miss patterns depend on input data and prior state. A cache miss can increase execution time by 10–100 cycles per memory access.
Compiler optimizations: Modern compilers reorder instructions, unroll loops, and inline functions in ways that make static analysis intractable.
Framework complexity: Inference engines (TensorFlow Lite Micro, PyTorch Mobile) are large codebases with dynamic dispatch and function pointers, making static path analysis infeasible.
The pragmatic approach to WCET for neural network inference is empirical: run the model on pathological inputs and measure the maximum observed latency, then add safety margin.
Step 1: Identify pathological inputs. For most feedforward networks with fixed-point or integer arithmetic, input data does not affect execution path—all inputs take the same path through the network. But for networks with dynamic layers (attention mechanisms, early exit branches, conditional execution), different inputs trigger different paths. Generate inputs that maximize the computational path (e.g., inputs that trigger all branches, avoid early exits, or produce the largest intermediate activations).
Step 2: Measure latency distribution. Run inference on thousands of test inputs and log latency for each. Plot the distribution. The maximum observed latency is your empirical WCET estimate.
Step 3: Add safety margin. The maximum observed latency is not guaranteed to be the true WCET—you may not have tested the input that triggers the worst case. Industry practice for safety-critical systems is to add 20–50% margin above the observed maximum. If measured maximum is 80 ms, WCET estimate is 80 × 1.3 = 104 ms.
Step 4: Verify on target hardware. WCET depends on the specific hardware configuration—clock speed, cache size, memory latency. Measure on the actual deployment target, not a development board with different specs.
For hard real-time applications (automotive, industrial control, medical devices), you may need provable WCET guarantees. This requires:
Using a certified WCET analysis tool (e.g., AbsInt aiT, Rapita RVS) that supports your processor.
Restricting the network to fixed-path architectures (no dynamic control flow).
Disabling caches or using cache locking to make memory latency deterministic.
Disabling interrupts during inference or running inference at highest priority.
These restrictions are expensive—fixed-path architectures limit model expressiveness, cache locking limits performance, and disabling interrupts increases system latency for other tasks. They're justified only when hard real-time guarantees are mandatory.
For soft real-time applications (most embedded AI deployments), empirical WCET with margin is sufficient. Test thoroughly, measure the 99.9th percentile latency, and add margin. If the application can tolerate occasional deadline misses, this is the right balance between rigor and practicality.
The Latency-Accuracy Tradeoff
Smaller models run faster but produce worse predictions. Larger models produce better predictions but run slower. The latency-accuracy tradeoff is the central design constraint for embedded AI.
Consider three variants of a gesture recognition model:
Model A: 2 million parameters, 80 million MACs, 92% accuracy
Model B: 800,000 parameters, 35 million MACs, 89% accuracy
Model C: 300,000 parameters, 12 million MACs, 84% accuracy
On a Cortex-M4 at 100 MHz sustaining 60 MMAC/s:
Model A: 1,333 ms latency
Model B: 583 ms latency
Model C: 200 ms latency
If your application has a 250 ms deadline, Model A fails, Model B fails, and Model C passes. But Model C's 84% accuracy may be below your application's requirement (say, 88% minimum). You're caught between latency and accuracy constraints.
The solution is to search the model architecture space for designs that lie on the Pareto frontier—models where no other model is both faster and more accurate. Model B is not on the Pareto frontier because Model C is faster (though less accurate) and Model A is more accurate (though slower). But if there exists a Model D with 500,000 parameters, 25 million MACs, and 90% accuracy, it Pareto-dominates Model C (same latency, better accuracy) and might be the optimal choice.
How do you find Pareto-optimal models? Two approaches:
Manual architecture search: Start with a well-known efficient architecture (MobileNetV2, EfficientNet-Lite) and scale it. Reduce input resolution, reduce channel count (width multiplier), reduce depth (number of layers). Measure latency and accuracy for each configuration. Plot them on a latency-accuracy graph and identify the Pareto frontier.
Neural Architecture Search (NAS): Use automated search algorithms to explore architecture space. NAS tools like Once-for-All (OFA), ProxylessNAS, or MnasNet search over architectures optimized for specific hardware constraints. You specify the target device and latency budget, and NAS proposes architectures that maximize accuracy subject to those constraints.
NAS is expensive—it requires GPU-hours to GPU-days to search the architecture space—but it often finds architectures that manual search misses. For high-volume products where even a 1% accuracy improvement justifies the search cost, NAS is worth it. For low-volume or one-off deployments, manual scaling is sufficient.
The latency-accuracy tradeoff is not negotiable. You cannot have both the best accuracy and the lowest latency. You must decide which constraint is binding—is latency a hard ceiling (the model must run in under 100 ms), or is accuracy a hard floor (the model must achieve at least 90%)? The binding constraint determines which models are viable.
A Worked Example: Evaluating Three Model Candidates for a 50ms Deadline
You are designing a gesture recognition system for a wearable fitness tracker. The system uses a 3-axis accelerometer sampled at 100 Hz. Gestures are classified every 1 second (100 samples). The application requires:
Latency: ≤50 ms from sample window complete to gesture output
Accuracy: ≥88% on a validation set of 20 gestures
Power: <5 mA average (the tracker runs on a coin cell)
You have three candidate models:
Model A: 1D CNN
Architecture: 4 convolutional layers + 2 fully connected layers
Parameters: 120,000 (480 KB as float32, 120 KB as int8)
Operation count: 18 million MACs
Measured accuracy: 91.2%
Model B: LSTM
Architecture: 2 LSTM layers (64 units each) + 1 fully connected layer
Parameters: 85,000 (340 KB as float32, 85 KB as int8)
Operation count: 14 million MACs (per time step, 100 steps total = 1.4 billion MACs)
Measured accuracy: 93.5%
Model C: Depthwise Separable CNN
Architecture: 3 depthwise separable blocks + 1 fully connected layer
Parameters: 45,000 (180 KB as float32, 45 KB as int8)
Operation count: 8 million MACs
Measured accuracy: 88.9%
Your target hardware is the nRF52840 (Cortex-M4 at 64 MHz with FPU). From prior profiling, you know the device achieves 50 MMAC/s sustained throughput for int8 inference with CMSIS-NN optimized kernels.
Latency analysis:
Model A: 18M / 50M = 360 ms. Fails the 50 ms deadline.
Model B: 1.4B / 50M = 28,000 ms = 28 seconds. Catastrophic failure. LSTMs are extremely expensive for long sequences on microcontrollers.
Model C: 8M / 50M = 160 ms. Fails the 50 ms deadline.
None of the models meet the latency requirement at 64 MHz. But the nRF52840 can run at 64 MHz in low-power mode or temporarily boost to 128 MHz for inference bursts (at higher power cost).
At 128 MHz, sustained throughput doubles to ~100 MMAC/s:
Model A: 18M / 100M = 180 ms. Still fails.
Model C: 8M / 100M = 80 ms. Still fails.
You realize the deadline is too aggressive for any model on this hardware without further optimization. You revisit the requirement with the product team and negotiate: latency can be up to 100 ms if accuracy is at least 88%.
At 128 MHz with the 100 ms deadline:
Model C: 80 ms latency, 88.9% accuracy. Passes both constraints.
Power analysis:
Running at 128 MHz increases active current from 15 mA (64 MHz) to 28 mA (128 MHz). Inference takes 80 ms. The device sleeps for 920 ms between classifications.
Average current: (28 mA × 0.08 s + 0.002 mA × 0.92 s) / 1 s = 2.24 mA + 0.002 mA = 2.24 mA.
The 5 mA budget allows for 2.24 mA inference plus 2.76 mA for sensor acquisition and communication. This is tight but feasible.
Conclusion:
Model C is the only viable option. Models A and B fail the latency constraint even with clock boosting. Model B (LSTM) is disqualified entirely—its 28-second latency is unusable on this hardware. Model C meets the negotiated requirements: 80 ms latency (under 100 ms), 88.9% accuracy (above 88%), and 2.24 mA average power (under 5 mA).
If the product team cannot accept the relaxed 100 ms deadline, the hardware is insufficient. You would need to:
Upgrade to a faster processor (e.g., STM32H7 at 480 MHz, which would achieve 400+ MMAC/s and bring Model C's latency to 20 ms), or
Add a hardware accelerator (e.g., Cortex-M55 + Ethos-U55 NPU), or
Further reduce the model (fewer layers, fewer channels, lower accuracy).
This is the compute constraint evaluation workflow. Theoretical analysis predicted feasibility. Profiling revealed the real bottleneck. Constraint relaxation (100 ms instead of 50 ms) made deployment viable. Without this analysis, you would have discovered the failure only after manufacturing thousands of units.
When You Cannot Meet the Latency Budget
If none of the available models meet the latency requirement, you have four options:
Option 1: Upgrade the hardware. Select a faster processor, add a hardware accelerator, or move to an application processor with NEON SIMD. This increases cost and power but solves the compute constraint definitively.
Option 2: Reduce the model's operation count. Shrink the architecture—fewer layers, fewer channels, lower input resolution. Measure the accuracy cost. If the accuracy floor is still met, deploy the smaller model.
Option 3: Relax the latency requirement. Negotiate with the product team. If the requirement was 50 ms because "that feels responsive," but 100 ms is actually acceptable, the constraint problem may disappear.
Option 4: Reject AI for this application. If no model meets both the accuracy and latency requirements on feasible hardware, AI is the wrong solution. Use a classical algorithm instead—a hand-tuned threshold-based classifier, a decision tree, or a finite state machine.
The compute constraint is unforgiving. Unlike memory, where you can swap to external storage at a performance cost, there is no workaround for insufficient compute. Either the processor can execute the operations in time, or it cannot. Optimization—SIMD, quantization, hardware acceleration—buys you 2–10× improvement. If you need 100× improvement, you've chosen the wrong hardware or the wrong model.
That's why this chapter follows memory in the diagnostic sequence. If the model fits in memory but cannot run fast enough, you know immediately that you need a faster processor, a smaller model, or a relaxed deadline. The power constraint (Chapter 7) examines what happens when the compute is fast enough but consumes too much energy.

---

*[← Chapter 5](./chapter-05-memory.md) | [Table of Contents](../README.md) | [Chapter 7 →](./chapter-07-power-and-energy.md)*