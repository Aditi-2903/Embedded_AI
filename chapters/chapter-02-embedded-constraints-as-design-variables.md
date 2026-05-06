# Chapter 2 — Embedded Constraints as Design Variables
*What the datasheet tells you, and what you have to figure out yourself.*

You already know how to read a microcontroller datasheet. You've decoded memory maps, calculated whether an ADC can sample fast enough, and figured out if sleep mode current will get you through a year on a coin cell. Those questions haven't changed. But when AI enters the system, you start asking different questions about the same specifications.

The nRF52840 datasheet lists 256 KB of RAM. That number meant one thing when you were buffering sensor data and running a PID controller. It means something different when you're storing activation tensors for a neural network. The STM32F4's 100 MHz clock speed looked adequate for your control loop. Does it look adequate when you need to multiply 128×128 matrices? The ESP32's sleep current of 10 µA was fine for a sensor that woke up once an hour. Is it fine when inference runs every five seconds?

The specifications are the same. The hardware is the same. What's changed is the translation from datasheet numbers to design constraints. This chapter teaches that translation—how to take the hardware you understand and ask the questions that AI integration demands.

<!-- → TABLE: Side-by-side comparison of traditional embedded constraints vs. AI-driven constraints for the same hardware specs — columns: specification (RAM, clock speed, power), traditional question asked, AI-driven question asked -->

## Memory: Where Things Live and Why It Matters

A neural network is a data structure. It has weights—millions of floating-point numbers arranged in matrices. It has biases, layer configurations, intermediate buffers. All of it must exist somewhere in memory. The first question for any AI deployment is: does it fit?

But "fit" isn't yes or no. Embedded memory is partitioned, and where data lives determines how you can use it. A model might fit in total memory and still fail because its runtime activations exceed the fast memory you actually have available during inference.

<!-- → INFOGRAPHIC: Memory hierarchy diagram for typical microcontroller showing flash (1 MB, slow, non-volatile, stores weights), on-chip SRAM (256 KB, fast, volatile, stores activations), and optional external PSRAM (32 MB, slower than on-chip, power-hungry) with arrows showing data flow during inference -->

Look at the nRF52840—a Bluetooth Low Energy microcontroller popular in wearables. The datasheet says 1 MB of flash, 256 KB of RAM. That sounds generous for embedded AI. Enough to store a model, enough to run it. But you have to interpret what those numbers mean.

Flash is non-volatile. It keeps data when power dies, which makes it right for storing firmware and model weights. But flash is read-only during execution—you can't write to it from running code without slow erase cycles that wear out the memory. Flash is also slower than RAM, often requiring wait states at high clock speeds. For AI, flash is where weights live. A model with 300,000 parameters stored as 32-bit floats needs 1.2 MB. That exceeds the nRF52840's 1 MB flash before you even account for the rest of your firmware.

RAM—SRAM on most microcontrollers—is volatile, fast, writable. It holds variables, stack, heap, the buffers that change during execution. For AI, RAM is where activations live. When you run inference on an image, each layer produces an output tensor that becomes input to the next layer. Those tensors must exist in memory at least for the duration of inference. The size of the largest intermediate tensor, plus scratch buffers the inference engine needs, determines minimum RAM.

The nRF52840's 256 KB of RAM is shared. Your application firmware takes some. The Bluetooth stack takes 20–40 KB depending on configuration. System heap takes some. If your model's activations need 180 KB and Bluetooth takes 30 KB, you have 46 KB left for everything else. That's workable if you're careful. If activations need 200 KB, you fail the RAM constraint even though the model "fits" in total memory.

<!-- → CHART: Stacked bar chart showing RAM allocation breakdown for nRF52840 — total 256 KB divided into segments: Bluetooth stack (30 KB), activations (180 KB), firmware/heap/stack (46 KB remaining). Include a second bar showing the failed case where activations need 200 KB, visually exceeding total capacity -->

External memory changes the picture. Some microcontrollers support external SRAM or DRAM through parallel or serial interfaces. The ESP32-S3 supports external PSRAM up to 32 MB. That dramatically increases available memory, but at a cost. External memory is slower than on-chip SRAM. Accessing it burns more power because data crosses an external bus. If your activations live in external PSRAM, every layer's computation reads from external memory and writes back to external memory. Latency goes up. Energy consumption goes up.

The question isn't "does the model fit?" The question is "does the model fit in the right kind of memory, and does the partitioning leave headroom for the rest of the system?"

Here's what you need. Two numbers from the model: weight size (determines flash usage) and activation buffer size (determines RAM usage). Three numbers from the hardware: flash capacity, on-chip RAM capacity, external memory if available. One number from the system: how much memory is already spoken for.

The calculation is direct. For weights: flash required equals number of parameters times bytes per parameter. A model with 500,000 parameters stored as 32-bit floats needs 2 MB. Stored as 8-bit integers after quantization, it needs 500 KB. If your target has 1 MB of flash and 200 KB is consumed by application firmware, you have 800 KB available. The float32 model doesn't fit. The int8 model does.

For activations, it depends on network architecture. A feedforward network needs memory for the largest layer's output plus scratch buffers. A convolutional network needs the current layer's output and sometimes the previous layer's output if they're used together. The exact requirement depends on how the inference engine allocates memory—whether it reuses buffers or allocates fresh for each layer.

TinyML frameworks usually provide a memory profiler. TensorFlow Lite for Microcontrollers has a function that returns the required tensor arena size—the working memory for inference. If that number exceeds available RAM, the model doesn't deploy.

Here's what this looks like. You're evaluating the ESP32-S3 for keyword spotting. The datasheet lists 512 KB on-chip SRAM, support for 32 MB external PSRAM. Your model has 80,000 parameters—320 KB as float32, 80 KB as int8—and needs 240 KB of activation memory.

Weights fit comfortably. The ESP32-S3 has 8 MB of flash. Even float32 uses only 320 KB. Activations are the binding constraint. With 512 KB on-chip SRAM, 240 KB for activations leaves 272 KB for everything else. But if the RTOS, WiFi stack, and application logic already consume 150 KB, you have 122 KB of headroom. Tight but feasible. If the system also needs large buffers for audio processing—the model runs on spectrograms—those buffers push SRAM over the limit and activations spill into external PSRAM.

Now the design choice is explicit. Accept the latency and power cost of external memory access, or reduce the model's activation footprint. Use a smaller architecture. Quantize activations. Apply layer fusion to shrink intermediate tensors.

This is what "memory as a design variable" means. The constraint isn't a wall. It's a trade-off surface where you balance model size, inference speed, power, system complexity.

## Compute: From Clock Speed to Operations Per Second

A microcontroller's clock speed tells you cycles per second. It doesn't tell you neural network operations per second. That translation—MHz to multiply-accumulates per second—is the second constraint you must quantify.

The gap exists because different operations take different numbers of cycles, and not all cycles are compute cycles. Memory access consumes cycles. Pipeline stalls consume cycles. Interrupt handling consumes cycles. Effective compute throughput—the rate at which you execute the multiply-accumulate operations that dominate inference—depends on instruction set, pipeline architecture, dedicated hardware for arithmetic, and how well inference code maps to those resources.

<!-- → TABLE: Three-column comparison of compute capabilities for STM32F411 (Cortex-M4, 100 MHz, 60-70 MMAC/s sustained), STM32H7A3 (Cortex-M7, 280 MHz, 180-220 MMAC/s sustained), Raspberry Pi Zero 2 W (Cortex-A53 quad-core, 1 GHz, >5 GOPS sustained int8) — showing theoretical peak, realistic sustained throughput, and inference time for 50M MAC model -->

Consider three targets: STM32F411 (ARM Cortex-M4 at 100 MHz), STM32H7A3 (Cortex-M7 at 280 MHz), Raspberry Pi Zero 2 W (Cortex-A53 quad-core at 1 GHz). All three get called "embedded" in some contexts, but their compute capabilities differ by orders of magnitude.

The STM32F411 is mid-range. Its Cortex-M4 includes a single-precision floating-point unit that can do one multiply-accumulate per cycle when data is in registers. At 100 MHz, that's 100 million MACs per second theoretical peak. But that's peak—achieved only when the pipeline is full and all data is cached. In practice you lose cycles to memory access, instruction fetch, pipeline bubbles. Realistic sustained throughput for inference is 60–70 MMAC/s with optimized code.

The Cortex-M4 also has SIMD via ARM DSP extensions, which let it do multiple 8-bit or 16-bit integer operations in parallel. For quantized networks using 8-bit arithmetic, SIMD can double or quadruple throughput versus scalar operations. But using SIMD requires careful optimization. It's not automatic.

The STM32H7A3 runs at 280 MHz with deeper pipelines, more aggressive prefetching, larger caches. Theoretical peak is 280 MMAC/s for single-precision floats. Sustained is typically 180–220 MMAC/s depending on memory patterns and cache efficiency. The H7 has bigger caches—16 KB L1 instruction, 16 KB L1 data, 512 KB L2—compared to the F4's 4 KB instruction cache and no data cache. That reduces the penalty for working with large structures like network weights.

The Raspberry Pi Zero 2 W is a different class entirely. Linux-capable application processor. Four Cortex-A53 cores with out-of-order execution, NEON SIMD for vector ops, clocks to 1 GHz. The A53 can issue multiple instructions per cycle. NEON can do 16 8-bit MACs per cycle per core. Theoretical peak exceeds 60 GOPS for 8-bit inference. Sustained is lower because of memory bandwidth and OS overhead, but still an order of magnitude higher than the STM32H7.

These differences determine how long inference takes. You have a convolutional network requiring 50 million MACs per inference. On the STM32F411 at 60 MMAC/s sustained, inference takes roughly 830 milliseconds. On the STM32H7 at 200 MMAC/s, it takes 250 ms. On the Pi Zero 2 W with optimized NEON code, maybe 20–30 ms.

If your application has a hard 100 ms latency requirement, the F411 fails immediately. The H7 fails if you can't sustain 200 MMAC/s—which you might not if memory accesses dominate. The Pi meets the requirement comfortably but costs you in power, complexity, boot time.

Here's the calculation. First, get the model's operation count—total MACs or FLOPs per inference. ML frameworks usually report this during compilation. If not, estimate from architecture. For a fully connected layer, operation count is input size times output size. For convolution, it's output height times output width times output channels times kernel height times kernel width times input channels.

Second, estimate the processor's sustained throughput for the relevant operation type. This needs profiling or published benchmarks. ARM provides CoreMark scores and EEMBC benchmarks. Vendor app notes sometimes include ML inference benchmarks for specific chips.

Third, calculate latency: total operations divided by sustained throughput.

<!-- → INFOGRAPHIC: Flow diagram showing the three-step compute constraint calculation — step 1: extract model operation count from framework profiler, step 2: find processor sustained throughput from benchmarks or datasheets, step 3: divide operations by throughput to get latency in seconds, with example numbers plugged in -->

A model with 120 million int8 MACs on an ESP32-S3—which claims 400 GOPS for 8-bit inference with acceleration—gives theoretical latency of 0.3 seconds. But that assumes perfect accelerator utilization, which requires the model compiled specifically for ESP32's AI extensions and all activations fitting the optimized data path. In practice, multiply theoretical latency by a utilization factor, often 0.5–0.7, to account for real inefficiencies.

The compute constraint is satisfied if calculated latency is less than your application's budget. If not, three options: faster processor, reduce the model's operation count, offload to hardware accelerator.

## Power: The Battery Life Equation

A neural network doesn't just consume time. It consumes energy. For battery-powered devices, energy is the binding constraint more often than memory or compute. A model that fits in memory and meets latency can still fail if it drains the battery too fast.

Power consumption isn't a single number. It's a profile varying with operating mode, clock speed, peripheral activity, duty cycle. A microcontroller running inference draws one current when the processor is active at full clock, a different current when peripherals are active but processor is idle, a third current in deep sleep. Total energy is the integral of that profile over time.

For AI, the power profile has three regimes: inference (processor active, high clock, high power), idle (processor sleeping, peripherals off, low power), and peripheral activity (sensors active, processor sleeping or reduced clock). The duty cycle—fraction of time in each regime—determines average power. Average power determines battery life.

<!-- → INFOGRAPHIC: Power profile timeline showing a typical 5-second cycle for wearable AI — 80 ms active at 5.3 mA (inference running), 4920 ms idle at 2.8 µA (deep sleep), with annotations showing how duty cycle determines average current and therefore battery life -->

Let's quantify this for a wearable fitness tracker using the nRF52840. It runs a neural network to classify activity from accelerometer data. Inference every 5 seconds. Between passes, the device sleeps.

The datasheet gives power specs for different modes:

- Active (CPU at 64 MHz, flash access, radio off): 5.3 mA
- Idle (CPU sleeping, RAM retained, RTC running): 2.8 µA
- System off (RAM not retained): 0.4 µA

Inference takes 80 ms measured on target. During those 80 ms, the processor is active drawing 5.3 mA. For the remaining 4920 ms of the 5-second cycle, idle mode draws 2.8 µA.

Average current is the time-weighted sum:

I_avg = (I_active × t_active + I_idle × t_idle) / (t_active + t_idle)

For this profile:

I_avg = (5.3 mA × 0.08 s + 0.0028 mA × 4.92 s) / 5 s = 0.088 mA

A 220 mAh coin cell sustains 0.088 mA for roughly 2500 hours—104 days. That exceeds the two-week target comfortably.

Now suppose inference latency doubles to 160 ms because you added a layer for better accuracy. Active time per cycle goes from 80 ms to 160 ms:

I_avg = (5.3 mA × 0.16 s + 0.0028 mA × 4.84 s) / 5 s = 0.172 mA

Battery life drops to 1279 hours, 53 days. Still acceptable.

But suppose the model now needs inference every second instead of every 5 seconds to catch short activities. With 160 ms inference per 1-second cycle:

I_avg = (5.3 mA × 0.16 s + 0.0028 mA × 0.84 s) / 1 s = 0.850 mA

Battery life drops to 259 hours, 11 days. Below the two-week threshold. Power constraint violated.

<!-- → CHART: Three-scenario comparison showing battery life degradation — scenario 1: 80 ms inference every 5s (104 days), scenario 2: 160 ms inference every 5s (53 days), scenario 3: 160 ms inference every 1s (11 days). Bar chart with battery life on Y-axis and scenarios on X-axis, with horizontal line marking the two-week target threshold -->

This isn't hypothetical. Deployed wearables fail this way. The failure isn't sudden—the device works correctly, just runs out of power faster than users expect. The constraint violation shows up as a UX problem: more frequent charging erodes the value proposition.

The power constraint is harder to satisfy than memory or compute because it couples to both. Reducing inference latency by running higher clock increases active current. Reducing memory usage by keeping activations in external RAM increases power because external access costs more energy than on-chip SRAM. Every optimization that helps one constraint can hurt another.

Power also has a peak constraint, not just average. Even if average power is acceptable, peak power can violate supply limits. When a processor runs inference, current spikes. If that spike exceeds what the supply or battery can deliver, voltage sags, the processor browns out, system resets. This is especially bad for coin cells, which have high internal resistance and can't sustain high draws even though total capacity is adequate.

A CR2032 coin cell has 220 mAh nominal capacity, but maximum continuous discharge current is typically 3 mA. If your inference draws 10 mA, battery voltage collapses under load and the system crashes. Solutions: larger battery, add a capacitor for peak current, reduce peak power by slowing clock or spreading computation over time.

The power equation you need: battery life equals battery capacity divided by average current. But to get average current, you must profile actual power consumption of inference on target hardware, measure duty cycle, account for all system activity—not just inference, but sensor power, communication overhead, background tasks.

## Reading Datasheets for AI

You have a model. You have an application with memory, compute, power, real-time requirements. You need hardware. The datasheet is your source of truth—but datasheets aren't written with AI in mind. Extracting relevant constraints requires knowing where to look and how to interpret marketing.

Start with memory specs. Flash capacity, SRAM capacity, external memory support. Flash must accommodate firmware plus model weights. SRAM must accommodate application heap, stack, RTOS overhead if applicable, inference activations. If the datasheet lists "total memory" without breaking down flash versus RAM, it's hiding something—usually that RAM is scarce.

<!-- → TABLE: Datasheet red flags for AI deployment — red flag example, what it actually means, what to look for instead. Examples: "total memory 1.25 MB" without breakdown (hiding scarce RAM, look for explicit flash/SRAM split), "AI acceleration support" with no benchmarks (marketing claim, look for GOPS numbers or published inference latency), "ultra-low power" with only sleep current (incomplete picture, look for power vs clock speed curves and real application notes) -->

Watch for memory region details. Some microcontrollers partition SRAM into tightly coupled memory (TCM)—fast and deterministic—and standard SRAM, which may have wait states or arbitration delays. TCM is ideal for activations. If TCM is listed separately, note its size. Often much smaller than total SRAM.

Examine compute specs. Clock speed is the start but not sufficient. Look for processor core type—Cortex-M0, M4, M7, A53—and check for FPU, DSP extensions, SIMD support. A Cortex-M4 with FPU can run float32 inference efficiently. A Cortex-M0+ without FPU will emulate floating-point in software—ten times slower.

If the datasheet claims "AI acceleration" or "neural network support," dig deeper. What does that mean? Some vendors include dedicated MAC units or vector processors for matrix operations. Others just mean the chip is fast enough to run inference in software. Look for benchmarks: GOPS, CoreMark scores, published inference latency for standard models like MobileNet. If no benchmarks, the "AI" claim is marketing.

Power specs need careful interpretation. Datasheets list current for different modes, but definitions vary by vendor. "Active current" might mean CPU at max clock with all peripherals enabled, or CPU running with flash access but peripherals off. "Sleep current" might include RAM retention or mean full power-down.

Find the power versus clock speed curve. Most microcontrollers can run at reduced clock to save power. If your inference doesn't need max performance, lower clock reduces both active current and energy per inference. But some chips have stepped profiles—cutting clock by 50% doesn't cut power by 50% because static leakage and peripheral power remain constant.

Look for real-world power numbers, not just active and sleep. App notes often include measured power for specific use cases. A note on "Battery Life for Bluetooth LE Sensors" might have the data you need if the sensor profile is similar to yours.

Check for errata. Every chip has bugs. Some errata affect power—"SRAM retention current higher than specified." Some affect peripherals. Some affect timing. Read the errata doc before committing to a platform. A bug that degrades sleep current by 10 µA can destroy battery life projections for ultra-low-power designs.

The workflow: identify flash and SRAM capacity, compare against model weight and activation requirements. Identify processor core and clock, look up benchmarks or estimate throughput. Calculate inference latency from operation count and throughput. Find active and sleep current, estimate average power from duty cycle. Calculate battery life from average power and capacity. Check real-time capabilities: cache, TCM, interrupt latency, worst-case timing.

If any constraint fails, the target isn't viable. If all pass with margin, it's a candidate. If constraints pass with no margin, revisit assumptions—measurements on real hardware often differ from datasheet projections by 20–30%.

## A Worked Example: Smart Agriculture Sensor

You're designing a sensor that monitors crop health with a camera and on-device classification. Requirements:

- Capture 96×96 RGB image every 10 minutes
- Run inference to classify healthy/diseased/pest-damaged
- Report over LoRaWAN once per hour
- Operate 6 months on 5000 mAh battery (solar not viable, tree canopy shading)

Three candidates:

**Option A: ESP32-S3**
- Dual-core Xtensa LX7 at 240 MHz
- 512 KB SRAM, 8 MB flash
- AI acceleration (vector instructions, 400 GOPS claimed)
- WiFi/Bluetooth (not needed but included)
- Active: 40 mA, deep sleep: 10 µA
- Cost: $4.50

**Option B: STM32L4R5**
- Cortex-M4 at 120 MHz with FPU
- 640 KB SRAM, 2 MB flash
- No AI acceleration
- Ultra-low-power design
- Active: 12 mA at 120 MHz, deep sleep: 0.4 µA
- Cost: $6.00

**Option C: Raspberry Pi Zero 2 W**
- Quad-core Cortex-A53 at 1 GHz
- 512 MB LPDDR2
- Linux-capable
- Active: ~150 mA idle, ~350 mA under load
- No true deep sleep
- Cost: $15.00

Your model: MobileNetV2-based classifier, 300,000 parameters (1.2 MB float32, 300 KB int8), 45 million MACs per inference, 200 KB activation memory.

<!-- → TABLE: Hardware comparison matrix for the three options — rows for each constraint dimension (flash requirement, RAM requirement, compute latency, power consumption, battery life, cost), columns for ESP32-S3, STM32L4R5, and Pi Zero 2 W, with pass/fail/marginal indicators and the actual calculated values -->

Memory analysis: all three have sufficient flash for quantized model (300 KB). All have sufficient RAM for activations (200 KB). Memory is not binding.

Compute analysis:

- ESP32-S3: with acceleration, 45M MACs at 400 GOPS claimed gives 112 ms theoretical, likely 200–300 ms in practice
- STM32L4R5: Cortex-M4 with FPU at 120 MHz sustains ~70 MMAC/s float32, giving 643 ms. For int8 with SIMD, ~140 MMAC/s, giving 320 ms
- Pi Zero 2 W: with NEON and four cores, sustained throughput for int8 can exceed 5 GOPS, under 10 ms

All three meet compute—inference every 10 minutes allows latency up to several seconds.

Power analysis—this is where they diverge:

Duty cycle: inference every 10 minutes for ~300 ms, idle rest of time. LoRaWAN once per hour is negligible.

ESP32-S3: active 0.3 s per 600 s cycle. Average current = (40 mA × 0.3 s + 0.01 mA × 599.7 s) / 600 s = 0.03 mA. Battery life = 5000 mAh / 0.03 mA = 166,667 hours = 19 years. The ultra-low sleep current dominates when duty cycle is this low.

STM32L4R5: active 0.32 s per 600 s. Average = (12 mA × 0.32 s + 0.0004 mA × 599.68 s) / 600 s = 0.0068 mA. Battery life = 735,000 hours = 84 years. Even lower sleep current gives multi-decade battery life. Unrealistic—self-discharge and LoRaWAN overhead will dominate.

Pi Zero 2 W: no true deep sleep. Best case idles at ~150 mA. Average ≈ 150 mA continuous. Battery life = 5000 mAh / 150 mA = 33 hours. Catastrophic failure of power constraint. Even aggressive optimization to 50 mA gives only 100 hours—four days, not six months.

<!-- → CHART: Battery life comparison for the three options on logarithmic scale — ESP32-S3 (19 years), STM32L4R5 (84 years), Pi Zero 2 W (33 hours), with the 6-month requirement marked as horizontal line. Visual makes clear that Pi fails catastrophically while other two vastly exceed requirements -->

Conclusion: ESP32-S3 and STM32L4R5 both meet all constraints. STM32 has better battery life and lower cost on paper, but ESP32's AI accelerator gives faster inference and headroom for future model updates. Pi is disqualified by power.

In practice you'd select based on secondary factors: development ecosystem, LoRaWAN library support, camera interface, whether you trust ESP32's claimed acceleration (needs profiling on real hardware).

This is constraint-driven evaluation. The numbers determine which hardware can deploy the model and which cannot. Chapter 3 shows you how to derive those model numbers—operation count, memory footprint—from the neural network architecture itself.

But first you must read a datasheet and extract constraints. The model is coming. The hardware is here. The question is: can they fit together?
