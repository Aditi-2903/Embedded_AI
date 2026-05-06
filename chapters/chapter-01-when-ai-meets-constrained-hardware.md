# Chapter 1 — When AI Meets Constrained Hardware
*When the model works and the hardware works, but the system fails anyway.*

A smart doorbell stopped working in November 2019. Not completely—it still rang when someone pressed the button, still sent alerts to your phone, still recorded video clips. But its battery, advertised to last six months, now died in forty-eight hours.

Nothing had broken. The hardware was identical to the unit that shipped three months earlier. The battery chemistry hadn't changed. What changed was a firmware update that added person detection—a neural network running on the doorbell's processor, analyzing each video frame to distinguish between a person at the door and a cat walking past.

The AI worked. It detected people. It reduced false alerts. But the system failed, and twelve thousand units came back in the first month.

Here's what happened. The engineering team tested the model in isolation on a development board. They verified its accuracy on a validation dataset. They confirmed the model could execute on the doorbell's ARM Cortex-M4 processor. What they didn't measure was how much energy the model consumed during continuous operation. They didn't calculate the average power draw when the processor woke up every few seconds to run inference. They didn't project how that power profile—active time plus sleep time—would translate into battery life under realistic deployment conditions.

The model was correct. The implementation was correct. But the integration—the act of putting that model on that hardware with that power budget—was never evaluated as a system-level constraint problem. The failure wasn't a bug. It was a design error: treating AI as software that happens to run on hardware, rather than as a design choice with measurable resource consequences.

This book exists because that error is common. When machine learning enters embedded systems, engineers trained in one domain make decisions that violate constraints they don't know how to quantify. ML engineers optimize for accuracy without checking whether the model fits in 256 KB of RAM. Embedded engineers allocate memory and clock cycles without understanding how neural network inference actually consumes those resources. The gap between what the model needs and what the hardware can provide becomes visible only after deployment—when batteries die too quickly, or inference takes too long, or the device crashes because activation buffers overflow SRAM.

The skill this book teaches is constraint evaluation. Not machine learning. Not embedded systems design. The thing in between: how to determine whether a trained model can run on constrained hardware before you deploy it, and if it can't, what trade-offs will make it fit.

You already know embedded systems or you know machine learning. This book assumes one and teaches you enough of the other to reason across the boundary. If you're an embedded engineer, you'll learn what neural networks cost in memory, compute, and power—not how to train them, but how to evaluate them as firmware artifacts. If you're an ML engineer, you'll learn how microcontroller datasheets translate into inference constraints—not how to design circuits, but how to read specifications and calculate whether your model fits.

The doorbell camera failed because the integration wasn't evaluated. This chapter shows you why that evaluation matters and what questions it must answer.

<!-- → IMAGE: Photo of the failed smart doorbell next to a coin cell battery, emphasizing the physical constraint of power supply -->

## What Embedded Actually Means

An embedded system is defined not by what it is but by what it cannot do.

A laptop can swap memory to disk when RAM fills up. A cloud server can provision more CPU cores when load spikes. A desktop GPU can draw three hundred watts from a wall outlet without consequence. An embedded system does none of these things. It has the memory it shipped with—no swap, no elasticity, no graceful degradation when you exceed capacity. It has the processor it was designed around—no autoscaling, no offloading to faster hardware. It has a power budget set by a battery or a supply rail, and exceeding that budget means the system stops working.

This is the first principle: resources are fixed, and exceeding them is not recoverable.

The word "embedded" gets used carelessly. A Raspberry Pi is embedded in the sense that it's small and runs Linux, but it has a gigabyte of RAM and can burst to 1.5 GHz when thermal headroom allows. An Arduino Nano has 32 KB of RAM and runs at 16 MHz with no operating system. Both get called embedded, but the constraints are different by two orders of magnitude. What they share is that their resources were allocated for a specific application. There's no headroom for general-purpose computing. You get what the designer gave you, and if you need more, you're working on the wrong hardware.

<!-- → TABLE: Comparison of "embedded" systems showing Raspberry Pi 4, Arduino Nano 33, STM32F4, and ESP32 with columns for RAM, Flash, Clock Speed, FPU presence, and typical power consumption -->

That matters for AI because neural networks are general-purpose artifacts. A face detection model doesn't know it's running on a battery-powered camera. It doesn't care whether your microcontroller has a floating-point unit or whether SRAM is shared with the Bluetooth stack. The model's resource requirements are set when it's trained—number of layers, size of weight matrices, precision of arithmetic. Those requirements don't negotiate. The embedded system, meanwhile, has its own fixed budget determined by cost targets, power limits, and physical constraints. Integration is what happens when you try to fit a fixed-demand artifact into a fixed-supply environment.

Consider a wearable health monitor. It runs on a coin cell battery that must last two weeks. It has 256 KB of SRAM, 1 MB of flash, and a 64 MHz ARM Cortex-M4 with no hardware floating-point unit. These aren't design choices you can revisit mid-project. They were locked in when someone decided the device had to be small enough to wear comfortably, cheap enough to hit a retail price point, and simple enough to manufacture at volume.

Now suppose you want to add heart arrhythmia detection using a recurrent neural network. The model exists. It's been trained, validated, achieves 94% accuracy on a test set. It requires 180 KB for weights, 60 KB for activation buffers during inference, and performs 22 million multiply-accumulate operations per pass. The model runs—you can execute it on a laptop, and it produces correct results.

But does it fit?

The memory budget is tight. Weights take 180 KB of flash, activations take 60 KB of RAM, which leaves 196 KB of RAM for everything else—sensor drivers, Bluetooth, power management, the application logic. That's workable if you're careful. The compute budget is worse. The processor does 64 million integer operations per second, but floating-point operations aren't native—they're emulated in software, which costs a factor of ten or more. Your 22 million multiply-accumulates become 220 million equivalent integer operations. At 64 million ops/sec, that's 3.4 seconds per inference pass.

The model must run every thirty seconds to catch arrhythmia events. That means the processor is active 3.4 seconds out of every 30-second window—an 11% duty cycle. If the processor draws 15 mA when active and 5 µA when sleeping, the average current is 1.7 mA. A 220 mAh coin cell sustains that for 130 hours. Five days, not two weeks.

<!-- → INFOGRAPHIC: Power consumption timeline showing 30-second cycles with 3.4 seconds active (processor running inference) and 26.6 seconds sleep, with current draw annotations and battery life calculation breakdown -->

The integration fails. Not because the model is wrong or the hardware is broken, but because the system—model plus hardware plus operating conditions—violates the power constraint. The device works exactly as designed and runs out of battery too quickly to be useful.

This has happened. Deployed wearables have failed this way. The failure mode isn't a crash or incorrect output. It's a user experience problem: the thing you bought runs out of power before you expect it to, and you stop using it.

Embedded constraints aren't bugs. They're the design space.

## Why AI Had to Move to the Edge

If embedded systems are constrained and neural networks are expensive, the obvious solution is: don't run AI on embedded systems. Send the data to a server. Let the cloud handle inference. Return the result over the network.

That architecture worked for the first wave of AI products. Alexa sends your voice to Amazon's servers. Google Photos does recognition in the cloud. Tesla uploads sensor logs for analysis when the car is parked and connected to WiFi.

But that architecture breaks for an entire class of applications, and those applications are why this book exists.

The failure modes are latency, privacy, connectivity, and cost. Each one is a constraint that better servers or faster networks cannot fix. They're fundamental to the application, not the implementation.

<!-- → INFOGRAPHIC: Four quadrants showing latency (industrial robot arm with 100ms window), privacy (medical wearable with shield icon), connectivity (agricultural sensor in remote field), and cost (parking sensor with dollar signs vs. data transmission costs) -->

Latency becomes a hard constraint when the time between sensing and action is measured in milliseconds. An industrial robotic arm inspects parts on a production line moving at two meters per second. A defect detection model must classify each part in the 100 milliseconds before it moves out of the inspection zone. If inference happens in the cloud, the round-trip includes sensor readout, network transmission, server queueing, inference, and result return. Even on a low-latency local network, that's 50 milliseconds or more. On a congested network or cellular backhaul, it balloons to hundreds of milliseconds. The inspection window closes before the answer arrives.

The application requires on-device inference not because the cloud is slow but because the speed of light is finite and the part won't wait.

Privacy becomes a hard constraint when the data cannot leave the device. Medical wearables that monitor cardiac rhythms handle protected health information. Even with user consent, transmitting raw physiological data over a network creates regulatory and liability risks that make cloud inference non-viable. The same applies to home security cameras, workplace monitoring, anything where the data itself is sensitive. On-device inference keeps data local. The model runs where the sensor is. Only the classification result—not the raw input—crosses the network boundary.

Connectivity becomes a hard constraint when the network is intermittent, expensive, or absent. Agricultural sensors in remote fields have cellular service only when a vehicle drives past. Underwater drones have no real-time uplink. Disaster response robots enter environments where infrastructure has failed. The model must run locally because the alternative isn't slower inference—it's no inference.

Cost becomes a hard constraint when network transmission dominates the operational budget. A parking sensor that reports occupancy state every hour can run for years on a coin cell. If that sensor had to stream video to a cloud server for vehicle detection, the cellular data cost and power consumption would make deployment economically infeasible. On-device inference shifts compute cost from operational expense—recurring network charges—to capital expense: one-time hardware cost.

These four constraints define where cloud AI stops working and edge AI becomes necessary. But necessity isn't the same as feasibility. An application might require on-device inference and still fail to achieve it because the embedded hardware cannot meet the model's resource demands.

That gap—between what the application requires and what the hardware can deliver—is the design space this book teaches you to navigate.

## The Four Constraints

Every AI integration on embedded hardware is governed by four resource constraints: memory, compute, power, and real-time performance. They're not independent. They interact. Satisfying one can violate another. Understanding how to quantify each constraint and how they couple is the foundational skill.

Memory determines what you can store. A neural network is a data structure. It has weights, biases, layer configurations, activation buffers. All of it must fit in the device's available memory. Embedded processors have two memory regions: flash (non-volatile, read-only during execution, abundant but slow) and SRAM (volatile, fast, writable, scarce). Flash stores the model weights. SRAM stores the activation tensors computed during inference. If activation memory exceeds available SRAM, the model cannot run, regardless of how much flash you have.

Compute determines how fast you can run. Inference is a sequence of operations—matrix multiplies, convolutions, activation functions. The time to complete one inference pass depends on the number of operations, the processor's throughput, and whether specialized hardware like a floating-point unit exists. Compute constraints show up as latency: how long from input to output. Some applications tolerate delays of hundreds of milliseconds. Others treat exceeding the deadline as failure.

Power determines how long you can run. Every operation consumes energy. Every memory access consumes energy. Energy comes from a battery, an energy harvester, or a supply with a limited budget. If the model's average power exceeds the available budget, battery life falls below acceptable thresholds, the thermal envelope is violated, or the supply cannot sustain the load. For battery-powered devices, power is often the binding constraint. A model that fits in memory and meets latency requirements can still fail if it drains the battery too quickly.

Real-time performance determines whether you meet application deadlines. Real-time doesn't mean fast. It means correctness depends on timing. A soft real-time system prefers to meet deadlines but tolerates occasional misses. A hard real-time system must meet deadlines with provable guarantees—missing one is system failure. Neural network inference introduces variability. Execution time can depend on input data, cache state, memory contention. For hard real-time applications, you must bound worst-case execution time, which is difficult when the model includes dynamic operations.

<!-- → TABLE: The four constraints showing constraint type, what it governs, failure mode, and typical embedded metrics. Rows for Memory (storage capacity, SRAM overflow, KB of SRAM), Compute (execution speed, missed deadlines, MMAC/s), Power (battery life, premature discharge, mA average current), Real-time (deadline guarantees, late results, ms worst-case) -->

These constraints couple. Quantizing weights from 32-bit floats to 8-bit integers reduces memory and speeds up inference but may degrade accuracy. Offloading to a hardware accelerator cuts latency and power but adds integration complexity and cost. Running inference less frequently reduces average power but increases latency between sensing and action.

Every design decision in embedded AI is a trade-off between these constraints. The designer who cannot quantify the trade-offs cannot make informed decisions.

## The Design Space

The design space for embedded AI is the intersection of two surfaces: what the model demands and what the hardware supplies. When they overlap, deployment is feasible. When they don't, you modify one or both.

Here's a concrete example. You're designing a voice interface that listens for a wake word—"Hey, device." The model is a convolutional network trained on audio spectrograms. It has 42,000 parameters, requires 180 KB of memory (weights plus activations), performs 1.8 million operations per inference, must complete within 100 milliseconds to feel responsive, and runs on a 1000 mAh battery that must last one week with continuous listening.

Now consider two hardware options.

Option A: ARM Cortex-M4 at 80 MHz, 256 KB SRAM, hardware floating-point unit, 20 mA active, 10 µA sleep, 1 MB flash, $3 per unit.

Option B: ARM Cortex-M0+ at 48 MHz, 32 KB SRAM, no FPU, 8 mA active, 2 µA sleep, 256 KB flash, $0.80 per unit.

Does the model fit on Option A?

Memory: 180 KB required, 256 KB available. Check.

Compute: With an FPU, the processor sustains roughly 80 million floating-point ops per second. The model needs 1.8 million, so inference takes about 22 milliseconds. Well within the 100 ms budget. Check.

Power: If the model runs continuously (once per 100 ms), the processor is active 22% of the time. Average current is 4.4 mA. Battery life is 227 hours, nine days. Close to the one-week target. Acceptable with optimization.

Option A works.

Does the model fit on Option B?

Memory: 180 KB required, 32 KB available. Fail immediately.

Even if you compressed the model to fit, compute likely fails too. Without an FPU, floating-point operations are emulated in software, reducing throughput by a factor of ten or more. The 1.8 million operations take 375 milliseconds, violating the 100 ms requirement.

Option B cannot run this model as-is.

<!-- → CHART: Side-by-side bar charts comparing Option A vs Option B across memory (required vs available), compute (latency vs budget), power (battery life vs target), and cost per unit -->

But Option B costs one-quarter as much. For a consumer product with thin margins and high volume, that cost difference is decisive. You face a choice: accept higher cost or modify the model. Modifying might mean quantizing weights to 8-bit integers, pruning layers, switching to a smaller architecture. Each change affects accuracy. If accuracy drops below the application threshold, the model becomes useless regardless of cost savings.

This is the design space. You have a model with fixed demands and hardware with fixed supply. You have an application with minimum requirements for accuracy, latency, battery life. Your job is to find a configuration where all constraints are satisfied simultaneously—or prove no such configuration exists and change the requirements.

## What This Book Teaches

This book teaches constraint evaluation in three parts.

Part I establishes the problem. You now know why embedded AI is a constrained optimization problem. Chapter 2 shows you how to quantify each constraint—how to read datasheets, calculate memory footprints, translate processor specs into inference throughput. Chapter 3 builds your ML literacy at the level needed to reason about inference—what happens during a forward pass, how architectures map to resources, which model classes suit which tasks.

Part II builds the diagnostic toolkit. Each chapter adds one constraint dimension. By the end, you can audit a proposed deployment and identify every constraint failure before writing code.

Part III extends the toolkit into model-aware design. The model becomes a design variable. You learn to select architectures for deployment constraints, compress models through quantization and pruning, navigate the toolchains that convert trained models to firmware.

The skill you'll have at the end: given a sensing task, application requirement, and hardware target, you can determine whether AI integration is feasible, select a deployment strategy, justify every decision against resource constraints, and document the trade-offs.

That doesn't make you a machine learning researcher. It doesn't make you an embedded architect. It makes you a systems integrator who can reason across both domains—the role the industry needs and currently lacks.

## What You Cannot Learn Here

This book teaches evaluation and integration, not invention. It assumes you can obtain a trained model—from a colleague, a model zoo, a vendor library. It doesn't teach you how to train from scratch, collect datasets, or design novel architectures. Those are ML skills taught in ML courses.

It also assumes working knowledge of embedded fundamentals—microcontroller programming in C, basic real-time concepts, how to read datasheets. If you lack that foundation, this isn't the right starting point.

What this book does teach—what no other book currently teaches—is how to evaluate the collision between ML models and embedded constraints, how to make integration decisions that survive deployment, and how to document those decisions so future engineers understand why the system is designed the way it is.

The doorbell camera failed because the integration wasn't evaluated. The model worked. The hardware worked. But the system—model on hardware under realistic conditions—did not work. The failure wasn't a bug. It was a design error: a decision made without quantifying resource consequences.

This book teaches you not to make that error. You start by learning to quantify constraints—which is what comes next.
