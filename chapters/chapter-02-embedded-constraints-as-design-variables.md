# Chapter 2: Embedded Constraints as Design Variables

You already know what a microcontroller datasheet looks like. You've read power consumption tables, decoded memory maps, and calculated whether a peripheral can meet a timing requirement. If you've designed embedded systems before, this chapter will feel familiar—but the questions you ask of the data will be different.
When you evaluated hardware for a sensor acquisition system, you cared whether the ADC could sample at 10 kHz, whether the UART buffer could handle peak data rates, and whether sleep mode current was low enough for a year of battery life. Those constraints still matter. But when AI enters the system, you add new questions: Can this processor multiply a 128×128 matrix in under 50 milliseconds? Can this memory hierarchy store 400,000 floating-point weights and still leave room for activation buffers? Can this device run inference every 100 milliseconds for a week on a coin cell?
The specifications haven't changed. The microcontroller's SRAM capacity, clock speed, and power consumption are the same whether you're running a PID controller or a neural network. What has changed is how you translate those specifications into design constraints. This chapter teaches that translation—how to take the hardware you already understand and ask the new questions that AI integration demands.
By the end of this chapter, you will be able to specify the memory, compute, power, and timing constraints of any embedded target in terms relevant to AI deployment. You will be able to read a datasheet and determine whether a proposed hardware platform can support a given model before you write a single line of code. And you will be able to compare two targets and explain precisely why one can run an AI model and the other cannot.
The skill is not new datasheet literacy. The skill is knowing what to look for when the application includes inference.
Memory: What Fits and Where It Lives
A neural network is a data structure. It has weights, biases, layer configurations, and intermediate buffers that must exist somewhere in the device's memory. The first constraint question for any AI deployment is: does it fit?
But "fit" is not a single yes-or-no question. Embedded memory is partitioned into regions with different characteristics, and where data lives determines how it can be used. A model that "fits" in total memory may still fail if its runtime activations exceed the available fast memory, or if its weights must be paged from external storage during inference.
Consider the nRF52840, a popular Bluetooth Low Energy microcontroller used in wearables and IoT devices. The datasheet lists 1 MB of flash and 256 KB of RAM. At first glance, that sounds generous for embedded AI—enough to store a moderately sized model and run inference. But the memory specification requires interpretation.
Flash is non-volatile storage. It retains data when power is removed, which makes it suitable for storing the firmware binary, including the model's trained weights. Flash is read-only during execution—you cannot write to it from running code without invoking special erase and write cycles that are slow and have limited endurance. Flash is also slower to access than RAM, typically requiring wait states when reading at high clock speeds. For AI deployment, flash is where the model weights live. A model with 300,000 parameters stored as 32-bit floats requires 1.2 MB—which exceeds the nRF52840's 1 MB flash budget before you account for the rest of the firmware.
RAM (SRAM on most microcontrollers) is volatile, fast, and writable. It holds the variables, stack, heap, and buffers that change during program execution. For AI inference, RAM is where activations live—the intermediate results computed as data flows through the network. When you run inference on an image classification network, each layer produces an output tensor that becomes the input to the next layer. Those tensors must exist in memory simultaneously, at least for the duration of the inference pass. The size of the largest intermediate tensor, plus any scratch buffers required by the inference engine, determines the minimum RAM requirement.
The nRF52840's 256 KB of RAM is shared between the application firmware, the Bluetooth stack (which can consume 20–40 KB depending on configuration), the system heap, and the inference activations. If your model's activation memory footprint is 180 KB, and the Bluetooth stack consumes 30 KB, you have 46 KB remaining for everything else. That's workable for a simple application, but tight. If the model's activations require 200 KB, the deployment fails the RAM constraint even though the model "fits" in total memory.
External memory complicates the picture. Some microcontrollers support external SRAM or DRAM via parallel or serial interfaces. The ESP32-S3, for example, includes integrated support for external PSRAM (pseudo-static RAM) up to 32 MB. This dramatically increases available memory, but at a cost: external memory is slower to access than on-chip SRAM, and accessing it consumes more power because data must traverse the external bus. If your model's activations live in external PSRAM, every layer's computation involves reading inputs from external memory and writing outputs back to external memory, which increases both latency and energy consumption.
The memory constraint question, then, is not "does the model fit?" but "does the model fit in the right kind of memory, and does the partitioning leave enough headroom for the rest of the system?"
To answer that question, you need two numbers from the model: the size of the weights (which determines flash usage) and the size of the activation buffers (which determines RAM usage). You need three numbers from the hardware: flash capacity, on-chip RAM capacity, and external memory capacity if available. And you need one number from the system: how much of that memory is already allocated to other functions.
The calculation is straightforward. For the weights:
Flash required = (number of parameters) × (bytes per parameter)
A model with 500,000 parameters stored as 32-bit floats requires 2 MB. Stored as 8-bit integers (quantized), it requires 500 KB. If the target has 1 MB of flash and 200 KB is already consumed by the application firmware, you have 800 KB available. The float32 model doesn't fit. The int8 model does.
For the activations, the calculation depends on the network architecture. For a feedforward network, you need memory for the largest layer's output tensor, plus any scratch buffers. For a convolutional network, you need memory for the current layer's output and sometimes the previous layer's output if they're used together. The exact requirement depends on the inference engine's memory allocation strategy—whether it reuses buffers or allocates fresh memory for each layer.
Many TinyML frameworks provide a memory profiler that reports activation memory usage for a given model. TensorFlow Lite for Microcontrollers, for example, includes a function that returns the required tensor arena size—the working memory needed for inference. If that number exceeds available RAM, the model doesn't deploy.
Here's what that looks like in practice. Suppose you're evaluating the ESP32-S3 for a keyword spotting application. The datasheet lists 512 KB of on-chip SRAM and support for up to 32 MB of external PSRAM. The model has 80,000 parameters (320 KB as float32, 80 KB as int8) and requires 240 KB of activation memory during inference.
The weights fit comfortably in flash—the ESP32-S3 has 8 MB of flash, and even the float32 version uses only 320 KB. The activations are the binding constraint. With 512 KB of on-chip SRAM, the 240 KB activation requirement leaves 272 KB for the rest of the system. But if 150 KB is already consumed by the RTOS, WiFi stack, and application logic, you have 122 KB of headroom—tight, but feasible. If the system also needs large buffers for audio processing (the keyword spotting model runs on audio spectrograms), those buffers push SRAM usage over the limit, and activations spill into external PSRAM.
The design choice is now explicit: accept the latency and power cost of external PSRAM access, or reduce the model's activation footprint by using a smaller architecture, quantizing activations, or applying layer fusion techniques that reduce intermediate tensor sizes.
This is what "memory as a design variable" means. The constraint is not a hard wall—it's a trade-off surface where you balance model size, inference speed, power consumption, and system complexity.
Compute: Throughput, Latency, and What the Processor Can Actually Do
A microcontroller's clock speed tells you how many cycles per second the processor executes. It does not tell you how many neural network operations per second it can perform. That translation—from MHz to multiply-accumulates per second—is the second constraint you must quantify.
The gap between clock speed and useful throughput exists because different operations take different numbers of cycles, and not all cycles are compute cycles. Memory access consumes cycles. Pipeline stalls consume cycles. Interrupt handling consumes cycles. The effective compute throughput—the rate at which you can execute the multiply-accumulate operations that dominate neural network inference—depends on the processor's instruction set, its pipeline architecture, whether it has dedicated hardware for floating-point or integer arithmetic, and how efficiently the inference code maps to those resources.
Consider three representative targets: the STM32F411 (ARM Cortex-M4 at 100 MHz), the STM32H7A3 (ARM Cortex-M7 at 280 MHz), and the Raspberry Pi Zero 2 W (ARM Cortex-A53 quad-core at 1 GHz). All three are "embedded" in the sense that they're used in resource-constrained applications, but their compute capabilities differ by orders of magnitude.
The STM32F411 is a mid-range microcontroller. Its Cortex-M4 core includes a single-precision floating-point unit (FPU) that can execute one floating-point multiply-accumulate (MAC) per cycle when data is in registers. At 100 MHz, that's a theoretical peak of 100 million MACs per second (100 MMAC/s). But that's peak throughput, achieved only when the pipeline is fully utilized and all data is in cache or registers. In practice, you lose cycles to memory access, instruction fetch, and pipeline bubbles. A realistic sustained throughput for inference is 60–70 MMAC/s for well-optimized code.
The Cortex-M4 also supports single instruction, multiple data (SIMD) operations via the ARM DSP extensions, which allow it to perform multiple 8-bit or 16-bit integer operations in parallel. For quantized neural networks that use 8-bit integer arithmetic, SIMD can double or quadruple throughput compared to scalar operations. But using SIMD requires careful code optimization—it's not automatic.
Now consider the STM32H7A3. Its Cortex-M7 core runs at 280 MHz and includes a double-precision FPU, deeper pipelines, and more aggressive prefetching and caching. Theoretical peak throughput is 280 MMAC/s for single-precision floats. Sustained throughput is typically 180–220 MMAC/s, depending on memory access patterns and cache efficiency. The H7 family also includes larger caches (16 KB L1 instruction, 16 KB L1 data, 512 KB L2) compared to the F4's smaller caches (4 KB instruction, no data cache), which reduces the performance penalty when working with large data structures like neural network weights.
The Raspberry Pi Zero 2 W is an entirely different class. It's a Linux-capable application processor with four Cortex-A53 cores, each capable of out-of-order execution, NEON SIMD extensions for vector operations, and clock speeds up to 1 GHz. The Cortex-A53 can issue multiple instructions per cycle, and NEON can execute 16 8-bit MACs per cycle per core. Theoretical peak throughput is over 60 billion operations per second (GOPS) for 8-bit integer inference. Sustained throughput is lower due to memory bandwidth limits and OS overhead, but still an order of magnitude higher than the STM32H7.
These differences matter because they determine how long inference takes. Suppose you have a convolutional neural network that requires 50 million MACs per inference pass. On the STM32F411 at 60 MMAC/s sustained, inference takes roughly 830 milliseconds. On the STM32H7 at 200 MMAC/s, it takes 250 milliseconds. On the Raspberry Pi Zero 2 W, with optimized NEON code, it might take 20–30 milliseconds.
If your application has a hard latency requirement of 100 milliseconds, the STM32F411 fails immediately. The STM32H7 fails if you cannot achieve 200 MMAC/s sustained (which you might not if memory accesses dominate). The Raspberry Pi meets the requirement comfortably—but at a cost in power, complexity, and boot time.
Here's the calculation you need to perform for any target. First, determine the model's operation count—the total number of MACs or floating-point operations (FLOPs) required per inference. Many ML frameworks report this during model compilation. If not, you can estimate it from the architecture: for a fully connected layer, the operation count is (input size) × (output size). For a convolutional layer, it's (output height) × (output width) × (output channels) × (kernel height) × (kernel width) × (input channels).
Second, estimate the processor's sustained throughput for the relevant operation type (float32 MACs, int8 MACs, etc.). This requires either profiling or consulting published benchmarks. ARM provides CoreMark scores and EEMBC benchmarks for their cores, which give rough guidance. Vendor application notes sometimes include ML inference benchmarks for their specific chips.
Third, calculate latency:
Inference latency (seconds) = (total operations) / (sustained throughput)
For example, a model with 120 million int8 MACs running on an ESP32-S3 (claimed 400 GOPS for 8-bit inference with acceleration) gives a theoretical latency of 0.3 seconds. But that assumes perfect utilization of the accelerator, which requires the model to be compiled specifically for the ESP32's AI extensions and all activations to fit in the optimized data path. In practice, you multiply the theoretical latency by a utilization factor (often 0.5–0.7) to account for real-world inefficiencies.
The compute constraint is satisfied if the calculated latency is less than the application's latency budget. If not, you have three options: choose a faster processor, reduce the model's operation count, or offload inference to a hardware accelerator.
Power: Average, Peak, and the Battery Life Equation
A neural network doesn't just consume time—it consumes energy. For battery-powered devices, energy is the binding constraint more often than memory or compute. A model that fits in memory and meets latency requirements can still fail if it drains the battery too quickly.
Power consumption in embedded systems is not a single number. It's a profile that varies with operating mode, clock speed, peripheral activity, and duty cycle. A microcontroller running inference draws one current when the processor is active at full clock speed, a different current when peripherals are active but the processor is idle, and a third current when the device is in deep sleep. Total energy consumption is the integral of that profile over time.
For AI deployment, the power profile has three regimes: inference (processor active, high clock, high power), idle (processor sleeping, peripherals off, low power), and peripheral activity (sensors active, processor sleeping or running at reduced clock). The duty cycle—the fraction of time spent in each regime—determines average power, and average power determines battery life.
Let's quantify this for a wearable fitness tracker using the nRF52840. The application runs a neural network to classify user activity (walking, running, sitting, cycling) from accelerometer data. Inference runs every 5 seconds. Between inference passes, the device sleeps.
The datasheet provides power specifications for different operating modes:
Active (CPU running at 64 MHz, flash access, radio off): 5.3 mA
Idle (CPU in sleep, RAM retention, RTC running): 2.8 µA
System off (RAM not retained, wake on reset): 0.4 µA
Inference takes 80 milliseconds (measured on target, not theoretical). During those 80 ms, the processor is active and draws 5.3 mA. For the remaining 4920 ms of the 5-second cycle, the device is in idle mode and draws 2.8 µA.
Average current is the time-weighted sum of each mode's current:
I_avg = (I_active × t_active + I_idle × t_idle) / (t_active + t_idle)
For this profile:
I_avg = (5.3 mA × 0.08 s + 0.0028 mA × 4.92 s) / 5 s
 I_avg = (0.424 mA·s + 0.0138 mA·s) / 5 s
 I_avg = 0.088 mA
A 220 mAh coin cell battery can sustain 0.088 mA for roughly 2500 hours, or 104 days. That exceeds the two-week target comfortably.
Now suppose the inference latency doubles to 160 milliseconds because you added another layer to improve accuracy. The active time per cycle increases from 80 ms to 160 ms:
I_avg = (5.3 mA × 0.16 s + 0.0028 mA × 4.84 s) / 5 s
 I_avg = 0.172 mA
Battery life drops to 1279 hours, or 53 days—still acceptable.
But suppose the model now requires inference every second instead of every 5 seconds to catch short-duration activities. With 160 ms inference time per 1-second cycle:
I_avg = (5.3 mA × 0.16 s + 0.0028 mA × 0.84 s) / 1 s
 I_avg = 0.850 mA
Battery life drops to 259 hours, or 11 days—below the two-week threshold. The power constraint is violated.
This is not a hypothetical calculation. Deployed wearables fail this way. The failure is not sudden—the device works correctly, it just runs out of power faster than users expect. The constraint violation manifests as a user experience problem: more frequent charging, which erodes the product's value proposition.
The power constraint is harder to satisfy than the memory or compute constraints because it couples to both. Reducing inference latency (compute constraint) by running at higher clock speed increases active current. Reducing memory usage by keeping activations in external RAM increases power because external memory accesses cost more energy than on-chip SRAM accesses. Every optimization that improves one constraint can degrade another.
Power also has a peak constraint, not just an average constraint. Even if average power is acceptable, peak power can violate supply limits. When a processor runs inference, current draw spikes. If that spike exceeds what the power supply or battery can deliver, the voltage sags, the processor brown-outs, or the system resets. This is especially problematic for coin cell batteries, which have high internal resistance and cannot sustain high current draws even though their total capacity is adequate.
For example, a CR2032 coin cell has a nominal capacity of 220 mAh, but its maximum continuous discharge current is typically 3 mA. If your inference routine draws 10 mA, the battery voltage collapses under load, and the system crashes. The solution is to use a larger battery, add a capacitor to supply peak current, or reduce peak power by slowing the clock or spreading the computation over time.
The power constraint equation you need is:
Battery life (hours) = Battery capacity (mAh) / Average current (mA)
But to get average current, you must profile the actual power consumption of inference on the target hardware, measure the duty cycle, and account for all system activity—not just the inference itself, but sensor power, communication overhead, and any background tasks.
Real-Time Constraints: Hard Deadlines and Worst-Case Execution Time
Latency is not the same as real-time performance. A system with 100 ms average latency is not the same as a system with a 100 ms worst-case latency guarantee. Real-time constraints care about the tail of the distribution, not the mean.
A real-time system is one where correctness depends on timing. A soft real-time system prefers to meet deadlines but tolerates occasional misses. A hard real-time system must meet deadlines with provable certainty—missing a deadline is a system failure, not a performance degradation.
Most embedded AI applications are soft real-time. A smart thermostat that classifies occupancy to adjust temperature can tolerate a delayed inference—the consequence is a slightly slower response, not a safety failure. But some applications are hard real-time. An industrial motor controller that uses AI to detect bearing faults must respond within a fixed time window or the fault progresses to catastrophic failure. A medical device that uses AI to detect cardiac arrhythmia must trigger an alert within a bounded time or the therapeutic window closes.
Neural network inference introduces variability that makes hard real-time guarantees difficult. Execution time can depend on:
Input data (some activations may short-circuit, some may trigger expensive branches)
Cache state (whether weights and activations are cached or must be fetched from main memory)
Memory contention (other tasks or DMA transfers competing for bus bandwidth)
Interrupt latency (if inference runs at normal priority and can be preempted)
For soft real-time applications, you measure average latency and ensure it's comfortably below the deadline with margin for variance. For hard real-time applications, you must bound the worst-case execution time (WCET) and prove that even the worst case meets the deadline.
WCET analysis for neural networks is an active research area with no settled practice. Traditional WCET analysis tools (like aiT or SWEET) model the processor's pipeline, cache, and memory system and compute the longest possible path through the code. But neural network inference code is large, complex, and often includes dynamic memory allocation and library calls that are difficult to analyze statically.
The pragmatic approach for hard real-time embedded AI is to constrain the problem:
Use a fixed-point or integer-only network with no dynamic control flow. Every layer executes the same operations regardless of input data.
Disable interrupts during inference or run inference at highest priority so it cannot be preempted.
Lock critical data (weights, activations) into cache or tightly coupled memory (TCM) to eliminate cache misses.
Measure WCET empirically by profiling inference on pathological inputs and adding safety margin (typically 20–50% over measured maximum).
Use a scheduling framework that enforces execution time budgets and kills tasks that overrun.
For example, suppose you have a vision-based inspection system that must classify parts on a conveyor belt moving at 1 meter per second. The inspection window is 200 ms. Inference must complete within that window, or the part leaves the field of view and the result is useless.
You profile inference on 10,000 test images and measure latency. The distribution shows a mean of 120 ms, a 95th percentile of 145 ms, and a maximum observed latency of 160 ms. Is 160 ms the WCET? Not necessarily—you may not have tested the input that triggers the worst case. You add 30% margin: 160 ms × 1.3 = 208 ms, which exceeds the 200 ms deadline.
The deployment fails the real-time constraint. You have three options: reduce the model's operation count to lower latency, increase the processor's clock speed, or redesign the inspection system to give inference more time (slow the conveyor, trigger earlier, etc.).
Real-time constraints are unforgiving. Average performance doesn't matter. Only the worst case matters. And if you cannot bound the worst case, you cannot deploy in a hard real-time application.
Reading Datasheets for AI Suitability
You have a model. You have an application with memory, compute, power, and real-time requirements. You need to select hardware. The datasheet is your source of truth—but datasheets are not written with AI deployment in mind, and extracting the relevant constraints requires knowing where to look and how to interpret marketing claims.
Start with memory specifications. Look for flash capacity, SRAM capacity, and external memory support. Flash capacity must accommodate the firmware plus model weights. SRAM capacity must accommodate the application heap, stack, RTOS overhead (if applicable), and inference activation buffers. If the datasheet lists "total memory" without breaking down flash vs. RAM, it's hiding something—usually that RAM is scarce.
Watch for memory region details. Some microcontrollers partition SRAM into tightly coupled memory (TCM), which is fast and deterministic, and standard SRAM, which may have wait states or arbitration delays. TCM is ideal for activations. If TCM is listed separately, note its size—it's often much smaller than total SRAM.
Next, examine compute specifications. Clock speed is the starting point, but not sufficient. Look for the processor core type (Cortex-M0, M4, M7, A53, etc.) and check whether it includes an FPU, DSP extensions, or SIMD support. A Cortex-M4 with FPU can run float32 inference efficiently. A Cortex-M0+ without FPU cannot—it will emulate floating-point in software, which is ten times slower.
If the datasheet claims "AI acceleration" or "neural network support," dig deeper. What does that mean? Some vendors include dedicated MAC units or vector processors that accelerate matrix operations. Others simply mean the chip is fast enough to run inference in software. Look for benchmark numbers: GOPS (giga-operations per second), CoreMark scores, or published inference latency for standard models like MobileNet or ResNet. If no benchmarks are provided, the "AI" claim is marketing.
Power specifications require the most careful interpretation. Datasheets list current consumption for different operating modes, but the definitions vary by vendor. "Active current" might mean CPU running at maximum clock with all peripherals enabled, or it might mean CPU running with flash access but peripherals off. "Sleep current" might include RAM retention or it might mean full power-down.
Find the power vs. clock speed curve. Most microcontrollers can run at reduced clock to save power. If your inference doesn't need maximum performance, running at a lower clock reduces both active current and the energy per inference. But some chips have stepped power profiles—reducing clock by 50% doesn't reduce power by 50% because static leakage and peripheral power remain constant.
Look for real-world power numbers, not just active and sleep. Application notes often include measured power for specific use cases. A note titled "Battery Life Estimation for Bluetooth LE Sensors" might include the data you need to estimate inference power if the sensor acquisition profile is similar to your application.
Finally, check for errata. Every silicon chip has bugs. Some errata affect power consumption (e.g., "SRAM retention current higher than specified"), some affect peripherals, some affect timing. Read the errata document before committing to a platform. A bug that degrades sleep current by 10 µA can destroy battery life projections for ultra-low-power designs.
Here's the workflow:
Identify flash and SRAM capacity. Compare against model weight size and activation memory requirement.
Identify processor core and clock speed. Look up published benchmarks or estimate throughput from architecture specs.
Calculate inference latency from model operation count and processor throughput.
Find active and sleep current. Estimate average power from duty cycle.
Calculate battery life from average power and battery capacity.
Check real-time capabilities: cache, TCM, interrupt latency, worst-case timing.
If any constraint fails, the target is not viable. If all constraints pass with margin, the target is a candidate. If constraints pass with no margin, revisit the assumptions—measurements on real hardware often differ from datasheet projections by 20–30%.
Comparing Targets: A Worked Example
You are designing a smart agriculture sensor that monitors crop health using a camera and on-device image classification. The system must:
Capture a 96×96 RGB image every 10 minutes
Run inference to classify healthy/diseased/pest-damaged
Report results over LoRaWAN once per hour
Operate for 6 months on a 5000 mAh battery pack (solar charging is not viable due to tree canopy shading)
You have three candidate platforms.
Option A: ESP32-S3
Dual-core Xtensa LX7 at 240 MHz
512 KB SRAM, 8 MB flash
AI acceleration (vector instructions, up to 400 GOPS claimed)
WiFi and Bluetooth (not needed but included)
Active current: 40 mA (CPU + AI accelerator)
Deep sleep: 10 µA
Cost: $4.50 per unit
Option B: STM32L4R5
ARM Cortex-M4 at 120 MHz with FPU
640 KB SRAM, 2 MB flash
No AI acceleration
Ultra-low-power design
Active current: 100 µA/MHz = 12 mA at 120 MHz
Deep sleep: 0.4 µA
Cost: $6.00 per unit
Option C: Raspberry Pi Zero 2 W
Quad-core ARM Cortex-A53 at 1 GHz
512 MB LPDDR2 RAM
Linux-capable, extensive software ecosystem
Active current: ~150 mA (idle), ~350 mA (CPU load)
Sleep modes: not designed for ultra-low-power
Cost: $15.00 per unit
You have a MobileNetV2-based classifier with 300,000 parameters (1.2 MB as float32, 300 KB as int8) and requires 45 million MACs per inference. Profiled activation memory is 200 KB.
Memory analysis:
All three platforms have sufficient flash for the quantized model (300 KB). All have sufficient RAM for activations (200 KB). Memory is not the binding constraint.
Compute analysis:
ESP32-S3: With AI acceleration, 45 million MACs at 400 GOPS claimed throughput gives 112 ms theoretical, likely 200–300 ms in practice depending on how well the model maps to the accelerator.
STM32L4R5: Cortex-M4 with FPU at 120 MHz sustains roughly 70 MMAC/s for float32. Inference takes 45M / 70M = 643 ms. For int8 with SIMD, throughput doubles to ~140 MMAC/s, giving 320 ms.
Raspberry Pi Zero 2 W: With NEON SIMD and four cores, sustained throughput for int8 inference can exceed 5 GOPS. Inference takes under 10 ms.
All three meet the compute constraint comfortably—inference once every 10 minutes allows latency up to several seconds without violating the application timing.
Power analysis:
This is where the platforms diverge.
The duty cycle is: inference every 10 minutes for ~300 ms (ESP32 or STM32), idle the rest of the time. LoRaWAN transmission is once per hour, negligible contribution to average power.
ESP32-S3:
 Active for 0.3 s per 600 s cycle = 0.05% duty cycle.
 I_avg = (40 mA × 0.3 s + 0.01 mA × 599.7 s) / 600 s = 0.03 mA.
 Battery life = 5000 mAh / 0.03 mA = 166,667 hours = 19 years.
Wait. That can't be right. The sleep current is 10 µA, not 0.01 mA—that's the same thing. Let me recalculate.
I_avg = (40 mA × 0.3 s + 0.01 mA × 599.7 s) / 600 s = (12 + 6) / 600 = 0.03 mA.
Actually, that is right. The ultra-low sleep current dominates the calculation when inference duty cycle is so low. This is why low-power embedded systems spend 99%+ of their time asleep.
STM32L4R5:
 Active for 0.32 s per 600 s cycle (slightly longer inference).
 I_avg = (12 mA × 0.32 s + 0.0004 mA × 599.68 s) / 600 s = 0.0064 mA + 0.0004 mA = 0.0068 mA.
 Battery life = 5000 mAh / 0.0068 mA = 735,000 hours = 84 years.
The STM32L4's lower active current and even lower sleep current give it a multi-decade battery life. This is unrealistic—battery self-discharge and LoRaWAN overhead will dominate.
Raspberry Pi Zero 2 W:
 The Pi doesn't have a true deep sleep mode. Best case, it idles at ~150 mA with reduced clock.
 I_avg ≈ 150 mA (continuous).
 Battery life = 5000 mAh / 150 mA = 33 hours.
The Raspberry Pi fails the power constraint catastrophically. Even if you could reduce idle power to 50 mA (aggressive optimization), battery life is only 100 hours—four days, not six months.
Conclusion:
Option A (ESP32-S3) and Option B (STM32L4R5) both meet all constraints. The STM32 has better battery life and lower cost on paper, but the ESP32's AI accelerator gives faster inference and more headroom for future model updates. The Raspberry Pi is disqualified by power consumption.
In practice, you'd select based on secondary factors: development ecosystem, LoRaWAN library support, camera interface availability, and whether you trust the claimed AI acceleration of the ESP32 (which requires profiling on real hardware).
This is the constraint-driven evaluation this chapter teaches. The numbers are not abstract—they determine which hardware can deploy the model and which cannot. When you move to Chapter 3, you'll learn how to derive those model numbers—the operation count, the memory footprint—from the neural network architecture itself.
But first, you must be able to read a datasheet and extract constraints. The model is coming. The hardware is here. The question is: can they fit together?

---

*[← Chapter 1](./chapter-01-when-ai-meets-constrained-hardware.md) | [Table of Contents](../README.md) | [Chapter 3 →](./chapter-03-ml-for-embedded-engineers.md)*