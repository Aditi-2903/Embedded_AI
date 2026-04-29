# Chapter 7: Power and Energy: The Efficiency Imperative

The model fits in memory. Inference completes in 45 milliseconds, well within the 100 millisecond deadline. You deploy to a battery-powered wildlife monitoring camera programmed to run inference every 30 seconds and transmit results once per hour. The system is spec'd for six months of operation on four AA batteries.

Three weeks into field testing, the batteries die. The deployed units report normal operation—correct classifications, no crashes, no errors—they just run out of power 22 times faster than calculated.

You retrieve a unit and profile it. Measured current draw during inference: 85 mA. Expected current draw: 40 mA. The model runs correctly but consumes twice the predicted power. When you scale that to continuous operation—inference every 30 seconds, 24 hours a day—the battery that should last 4,380 hours lasts 200 hours instead.

This is the power constraint. A model can fit in memory, meet latency requirements, and still fail in deployment because it drains the battery too quickly. For battery-powered and energy-harvesting systems, power is often the binding constraint—the one that determines whether the product is viable or whether the deployment fails in the field.

This chapter teaches you to estimate the energy cost of inference, determine whether a deployment strategy is compatible with a power budget, design inference scheduling strategies that maximize battery life, and profile power consumption on target hardware. By the end of this chapter, you will be able to evaluate the power constraint as rigorously as you evaluate memory and compute.

## Power vs. Energy: The Distinction That Matters

Power and energy are related but not interchangeable. Confusing them leads to incorrect calculations and failed deployments.

Power is the instantaneous rate of energy consumption, measured in watts (W) or milliwatts (mW). A device drawing 50 mW consumes energy at a rate of 50 millijoules per second.

Energy is the total amount of work done or consumed over time, measured in joules (J) or watt-hours (Wh). A device drawing 50 mW for 10 seconds consumes 0.5 joules of energy.

The relationship is:

Energy (J) = Power (W) × Time (s)

For embedded systems running on batteries, what matters is total energy consumption, not instantaneous power—unless instantaneous power exceeds what the battery or power supply can deliver (the peak power constraint).

A battery stores a fixed amount of energy. A 1000 mAh lithium-ion cell at 3.7V stores:

Energy = 1000 mAh × 3.7 V = 3.7 Wh = 13,320 J

If your device consumes energy at an average rate of 10 mW, battery life is:

Battery life = 13,320 J / 0.01 W = 1,332,000 seconds = 370 hours = 15.4 days

But if your device consumes 10 mW on average by running at 200 mW for 5% of the time and sleeping at 0.5 mW for 95% of the time, the calculation is different:

Average power = (200 mW × 0.05) + (0.5 mW × 0.95) = 10 + 0.475 = 10.475 mW

The battery life is nearly the same because average power is nearly the same. This is why duty cycling—running the processor at high power for short bursts, then sleeping—is fundamental to embedded AI power management.

The mistake embedded designers make is calculating power based on active current alone. A model that runs at 60 mA for 80 ms every 10 seconds does not consume 60 mA average. It consumes:

I_avg = (60 mA × 0.08 s + 0.005 mA × 9.92 s) / 10 s = 0.48 mA + 0.05 mA = 0.53 mA

That's 100× lower than the active current. The sleep current dominates average power when duty cycle is low.

The power constraint question is: given a battery capacity, an inference schedule, and measured active and sleep currents, how long does the battery last? If the answer is less than the application requirement, the power constraint fails.

## Dynamic and Static Power in AI Inference Workloads

Power consumption in digital circuits has two components: dynamic power and static power.

Dynamic power is the energy consumed by switching transistors—charging and discharging capacitive loads as logic gates toggle between 0 and 1. Dynamic power is proportional to:

Clock frequency (higher frequency = more transitions per second)

Supply voltage squared (higher voltage = more energy per transition)

Activity factor (fraction of transistors switching per cycle)

P_dynamic = α × C × V² × f

where α is the activity factor, C is the capacitance being switched, V is the supply voltage, and f is the clock frequency.

For AI inference, dynamic power dominates when the processor is active. Every multiply-accumulate operation charges and discharges transistors. Every memory access switches address and data lines. Clock frequency directly controls dynamic power—running at 200 MHz consumes roughly twice the dynamic power of running at 100 MHz, assuming voltage stays constant.

Static power (also called leakage power) is the energy consumed by transistors even when they're not switching—current that leaks through the gate oxide and junctions. Static power is nearly independent of clock frequency. It depends on:

Supply voltage (higher voltage = more leakage)

Temperature (higher temperature = more leakage, exponentially)

Process technology (smaller transistors = more leakage per area)

For modern microcontrollers at room temperature, static power is 1–10% of dynamic power when the processor is active. But when the processor is asleep (clock gated, most transistors off), static power dominates. Sleep current is almost entirely leakage.

This matters for AI inference because active and sleep power scale differently. Reducing clock frequency reduces dynamic power linearly but doesn't affect static power. If your inference workload is short bursts (10 ms active, 990 ms sleep), reducing clock frequency from 200 MHz to 100 MHz halves the active power but barely changes average power because sleep power dominates.

## The optimization strategy depends on duty cycle:

High duty cycle (>50% active time): Reduce clock frequency, reduce voltage (DVFS—dynamic voltage and frequency scaling), or offload to a more efficient accelerator.

Low duty cycle (<5% active time): Reduce sleep current (use deep sleep modes, disable peripherals, gate power rails), minimize active time (faster inference at higher clock, not slower inference at lower clock).

A common mistake is running inference at low clock speed to "save power." If inference takes 400 ms at 64 MHz but only 200 ms at 128 MHz, and sleep current is 5 µA, which is better?

At 64 MHz:

Active current: 20 mA for 400 ms

Sleep current: 0.005 mA for 9,600 ms

Average current: (20 × 0.4 + 0.005 × 9.6) / 10 = 0.8 + 0.048 = 0.85 mA

At 128 MHz:

Active current: 35 mA for 200 ms

Sleep current: 0.005 mA for 9,800 ms

Average current: (35 × 0.2 + 0.005 × 9.8) / 10 = 0.7 + 0.049 = 0.75 mA

Running at 128 MHz (higher clock, higher active power, shorter active time) saves 12% average power compared to 64 MHz. This is "race to sleep"—finish inference as quickly as possible and return to low-power sleep mode.

The crossover point depends on the ratio of active current to sleep current and the duty cycle. For systems that spend >90% of the time asleep, race to sleep almost always wins. For systems that are continuously active, reducing clock frequency saves power.

## Duty Cycling and Inference Scheduling

Duty cycling is the technique of alternating between active and sleep states to reduce average power consumption. For embedded AI, the duty cycle is determined by:

How often inference must run (the inference interval)

How long each inference takes (the inference latency)

What sleep mode the system uses between inferences

A wildlife camera that runs inference once every 60 seconds, where inference takes 150 ms, has a duty cycle of:

Duty cycle = 0.15 s / 60 s = 0.25%

The processor is active for 0.25% of the time and asleep for 99.75% of the time. If active current is 50 mA and sleep current is 10 µA, average current is:

I_avg = (50 mA × 0.0025) + (0.01 mA × 0.9975) = 0.125 mA + 0.01 mA = 0.135 mA

A 2000 mAh battery lasts 2000 / 0.135 = 14,800 hours = 617 days.

But if the camera must run inference every 5 seconds instead of every 60 seconds (to catch faster-moving animals), duty cycle increases to 3%:

I_avg = (50 mA × 0.03) + (0.01 mA × 0.97) = 1.5 mA + 0.01 mA = 1.51 mA

Battery life drops to 1324 hours = 55 days.

Increasing the inference rate by 12× reduces battery life by 11×. This is the inference scheduling constraint: more frequent inference costs energy proportional to the increase in duty cycle.

The design question is: how often must inference run to meet the application requirement? There are three scheduling strategies:

Periodic inference: Run inference at fixed intervals (every N seconds). This is simple and predictable. It works when the application has no external trigger—anomaly detection on continuous sensor streams, environmental monitoring, health tracking.

Event-driven inference: Run inference only when a low-power sensor detects an event of interest. A motion detector wakes the system, then the camera runs inference. A voice activity detector (VAD) wakes the system, then the keyword spotter runs. This dramatically reduces duty cycle for applications where events are rare—a camera that triggers on motion might run inference 10 times per day instead of 1440 times per day.

Adaptive inference: Vary the inference interval based on context. Run inference frequently when activity is detected, infrequently when the environment is stable. A predictive maintenance system might run inference every 10 seconds when vibration levels are high, but only every 10 minutes when vibration is within normal range. This balances responsiveness and power consumption.

Event-driven inference is the most power-efficient when applicable. The energy cost of the trigger mechanism (motion detector, VAD) must be lower than the energy cost of continuous inference. A motion detector that consumes 50 µA continuously saves energy if it avoids running inference that consumes 50 mA for 150 ms every 10 seconds—the motion detector uses 0.05 mA continuous, while periodic inference would use (50 × 0.015 + 0.01 × 9.985) / 10 = 0.075 + 0.1 = 0.175 mA average.

The workflow for duty cycle optimization is:

Determine the minimum inference interval that meets application requirements.

Measure inference latency on target hardware.

Calculate duty cycle and average current.

If average current exceeds the power budget, apply one of: reduce inference interval (if the application allows), reduce inference latency (faster model or faster processor), or use event-driven triggering.

## Energy Harvesting Constraints

Energy harvesting systems—powered by solar cells, piezoelectric generators, thermoelectric generators, or RF harvesting—have different power constraints than battery-powered systems. Instead of a fixed energy budget that depletes over time, energy harvesters have a power budget that must be balanced in steady state.

An energy harvesting system is viable if:

Average power consumption ≤ Average power generation

A solar-powered sensor with a 1 cm² solar cell generates roughly 10 mW in full sunlight (100 mW/cm² irradiance × 10% conversion efficiency). If the sensor consumes 5 mW on average, the system is energy-positive and can run indefinitely. If the sensor consumes 15 mW on average, the system is energy-negative and will deplete any storage capacitor or battery backup.

But energy harvesting is intermittent. Solar power drops to near-zero at night. Vibration harvesting depends on machine operation. RF harvesting depends on proximity to a transmitter. The system must store enough energy during high-harvesting periods to survive low-harvesting periods.

## For AI inference, this creates two constraints:

Peak power constraint: The instantaneous power draw during inference must not exceed the harvester's peak output. If the harvester produces 20 mW peak and inference draws 100 mW, the system cannot run inference directly from harvested power—it must accumulate energy in a storage capacitor, then discharge the capacitor during inference.

Average energy balance: Over a full harvesting cycle (e.g., 24 hours for solar), total energy consumed must not exceed total energy harvested. If the harvester generates 10 mWh per day and the system consumes 12 mWh per day, the system fails.

A worked example: A vibration-powered industrial sensor harvests 5 mW when the motor is running (16 hours per day) and 0 mW when the motor is off (8 hours per day). Total harvested energy per day:

E_harvested = 5 mW × 16 hours = 80 mWh

The sensor runs inference every 60 seconds. Inference takes 120 ms at 80 mA (3.3V supply). Energy per inference:

E_inference = 3.3 V × 0.08 A × 0.12 s = 0.0317 J = 0.0088 mWh

Inferences per day: 1440. Total inference energy: 1440 × 0.0088 = 12.7 mWh.

Sleep power: 3.3 V × 0.02 mA = 0.066 mW. Sleep time per day: 24 hours - (1440 × 0.12 s) / 3600 s = 24 - 0.048 = 23.95 hours. Sleep energy: 0.066 mW × 23.95 hours = 1.58 mWh.

Total energy consumed: 12.7 + 1.58 = 14.3 mWh.

The system consumes 14.3 mWh but harvests 80 mWh. Energy balance is positive by a factor of 5.6×. The system can run indefinitely with significant margin.

But if inference must run every 10 seconds instead of every 60 seconds, inferences per day increase to 8,640. Inference energy: 8,640 × 0.0088 = 76 mWh. Sleep energy remains roughly the same. Total: 77.6 mWh.

The system now consumes 77.6 mWh and harvests 80 mWh. Energy balance is positive by only 3%. This is marginal—harvester output varies with temperature, vibration amplitude, and mechanical aging. A 5% drop in harvesting efficiency causes the system to fail.

The energy harvesting constraint is unforgiving. Unlike battery-powered systems where you can ship with a larger battery, energy harvesting systems have fixed input power. If the system consumes more than it harvests, no amount of optimization will make it viable—you must reduce consumption, increase harvesting area, or reject the deployment.

## Event-Driven AI: Using Low-Power Sensors to Trigger Inference

A camera running continuous inference at 10 fps draws power continuously. A camera that runs inference only when motion is detected draws power in bursts. The difference in battery life can be 100× for applications where events are rare.

Event-driven AI uses a low-power trigger mechanism to wake the system and run inference. The trigger is always-on, but its power consumption is orders of magnitude lower than the AI model.

## Common trigger mechanisms:

Motion detectors: Passive infrared (PIR) sensors consume 50–200 µA and wake on motion. A wildlife camera with a PIR trigger runs inference only when an animal enters the field of view. If animals trigger the camera 20 times per day, the camera runs 20 inferences per day instead of 86,400 (one per second). Energy savings: 4,300×.

Voice activity detectors (VAD): Detect speech in audio without running a full keyword spotter. A VAD can be as simple as an energy threshold detector or as complex as a small neural network running on a low-power DSP. When the VAD detects speech, it wakes the main processor to run the keyword spotter. If speech occurs 5% of the time, energy savings are 20×.

Accelerometer interrupts: Configure an accelerometer to generate an interrupt when motion exceeds a threshold. A gesture recognition system runs inference only when the wearable detects movement, not continuously. If the user is stationary 80% of the time, energy savings are 5×.

Hierarchical inference: Run a tiny, low-power model continuously to detect events, then run a larger, more accurate model only when the first model triggers. A two-stage system might use a 10,000-parameter decision tree (1 ms inference, 2 mA) running every 100 ms to detect anomalies, then trigger a 500,000-parameter CNN (150 ms inference, 60 mA) only when the decision tree flags a potential anomaly. If anomalies occur 1% of the time, the two-stage system consumes 90% less power than running the CNN continuously.

The design workflow is:

Identify a low-power signal that correlates with the need for inference.

Measure the trigger mechanism's power consumption and false-positive rate.

Calculate average power with event-driven triggering: (trigger power × 100%) + (inference power × event rate × inference time).

Compare to continuous inference power.

If event-driven power is lower and the false-positive rate is acceptable, deploy event-driven inference.

The failure mode is high false-positive rates. If the motion detector triggers on wind, shadows, or small insects, the camera runs inference thousands of times per day instead of 20 times per day, and energy savings disappear. Trigger mechanisms must be tuned to the application's event statistics.

## Power Profiling: How to Measure Inference Energy Consumption

Estimating power consumption from datasheets is unreliable. Active current depends on clock speed, memory access patterns, and peripheral usage. Sleep current depends on temperature, enabled peripherals, and retention modes. The only way to know actual power consumption is to measure it.

The standard method is a current sense resistor in series with the power supply. You insert a small resistor (0.1–1 Ω) between the battery and the device, measure the voltage drop across the resistor, and compute current using Ohm's law. The voltage drop must be small enough not to affect device operation—a 0.1 Ω resistor with 100 mA current drops only 10 mV, which is negligible.

For dynamic measurements, you log voltage over time using an oscilloscope or a data acquisition system. Modern power profilers (e.g., Nordic Power Profiler Kit II, Keysight N6705C) measure current at kHz to MHz sampling rates, allowing you to see individual inference bursts.

A typical power profile for inference looks like:

Baseline: sleep current (5–50 µA)

Spike: sensor acquisition (10–50 ms at 5–20 mA)

Spike: inference execution (50–500 ms at 30–200 mA)

Return to baseline: sleep current

You integrate the area under the current curve to get total energy per inference:

Energy (J) = ∫ V(t) × I(t) dt

For a 3.3V device:

Energy (J) = 3.3 V × ∫ I(t) dt

If the current profile is a square wave (simplified assumption), the integral becomes:

Energy = V × I_active × t_active

For 80 mA active current, 150 ms active time:

Energy = 3.3 V × 0.08 A × 0.15 s = 0.0396 J = 39.6 mJ

If inference runs every 30 seconds, average power is:

P_avg = 0.0396 J / 30 s = 1.32 mW

A 2000 mAh battery at 3.3V stores 6600 mWh = 23,760 J. Battery life:

Life = 23,760 J / 1.32 mW = 18 million seconds = 208 days

But this assumes sleep current is zero, which is unrealistic. If sleep current is 10 µA:

Sleep power = 3.3 V × 0.01 mA = 0.033 mW

Sleep time per cycle: 30 s - 0.15 s = 29.85 s. Sleep energy: 0.033 mW × 29.85 s = 0.985 mJ.

Total energy per cycle: 39.6 mJ (inference) + 0.985 mJ (sleep) = 40.6 mJ.

Average power: 40.6 mJ / 30 s = 1.35 mW.

Battery life: 23,760 J / 1.35 mW = 17.6 million seconds = 204 days.

The sleep energy contribution is small (2.4%) because duty cycle is low. But for higher duty cycles or higher sleep currents, sleep energy can dominate.

Profiling reveals surprises. Peripherals left enabled during sleep, unintended wake-ups, GPIO leakage currents—all contribute to higher-than-expected power consumption. The wildlife camera that drained batteries in three weeks likely had one of these issues: a peripheral left active, a firmware bug that prevented deep sleep entry, or GPIO pins floating and drawing leakage current.

The profiling workflow is:

Connect a current sense resistor and measurement equipment.

Run the full application (sensor acquisition, inference, communication, sleep) in a realistic duty cycle.

Log current over multiple inference cycles (at least 10).

Compute energy per cycle and average power.

Calculate battery life: (battery capacity in Wh) / (average power in W).

If battery life is below requirement, identify the power-hungry component and optimize it.

## A Worked Example: Designing Inference Scheduling for Six-Month Battery Life

You are designing a remote agricultural sensor that monitors soil moisture and runs on two AA alkaline batteries. The system must last six months (4,380 hours) in the field. The sensor uses:

STM32L4 microcontroller (1 µA deep sleep, 12 mA active at 80 MHz)

Soil moisture sensor (500 µA when powered, 0 µA when off)

LoRaWAN radio (40 mA transmit, 15 mA receive, 1 µA sleep)

You have a neural network for anomaly detection in moisture patterns:

100,000 parameters (int8, 100 KB in flash)

15 million MACs per inference

Measured latency: 220 ms (profiled on target)

Active current during inference: 12 mA (measured)

Two AA alkaline batteries provide 3V and 2850 mAh capacity. Total energy:

E_battery = 3 V × 2850 mAh = 8550 mWh = 30,780 J

Target average power for six months:

P_avg = 30,780 J / (4380 hours × 3600 s/hour) = 30,780 J / 15.77 million s = 1.95 mW

Your task: design an inference schedule that keeps average power below 1.95 mW.

Option 1: Periodic inference every 10 minutes

Energy per inference cycle (10 minutes = 600 s):

Sensor acquisition: 3 V × 0.5 mA × 5 s = 7.5 mJ

Inference: 3 V × 12 mA × 0.22 s = 7.92 mJ

LoRa transmission: 3 V × 40 mA × 0.5 s = 60 mJ (once per hour, so 1/6 per cycle = 10 mJ per cycle)

Sleep: 3 V × 0.001 mA × 594.28 s = 1.78 mJ

Total per cycle: 7.5 + 7.92 + 10 + 1.78 = 27.2 mJ

Average power: 27.2 mJ / 600 s = 0.045 mW

This is far below the 1.95 mW budget. Battery life:

Life = 30,780 J / 0.045 mW = 684 million s = 7,903 days = 21.6 years

The constraint is not tight. You can run inference much more frequently if needed.

Option 2: Periodic inference every 1 minute

Energy per cycle (60 s):

Sensor: 7.5 mJ

Inference: 7.92 mJ

LoRa: 10 mJ (amortized per hour still 1/60 per cycle = 1 mJ per minute cycle... wait, let me recalculate)

If LoRa transmits once per hour and inference runs every minute, there are 60 inference cycles per LoRa transmission. LoRa energy per cycle:

E_LoRa_per_cycle = 60 mJ / 60 = 1 mJ

Energy per cycle:

Sensor: 7.5 mJ

Inference: 7.92 mJ

LoRa (amortized): 1 mJ

Sleep: 3 V × 0.001 mA × 51.78 s = 0.155 mJ

Total: 16.6 mJ per minute

Average power: 16.6 mJ / 60 s = 0.277 mW

Battery life: 30,780 J / 0.277 mW = 111 million s = 3.5 years

Still far below budget.

Option 3: Continuous inference (every 1 second)

This is likely unnecessary for soil moisture monitoring, but let's check the power budget:

Energy per cycle (1 s):

Sensor: 7.5 mJ

Inference: 7.92 mJ

LoRa (amortized): 60 mJ / 3600 = 0.0167 mJ

Sleep: 3 V × 0.001 mA × 0.28 s = 0.00084 mJ

Total: 15.44 mJ per second

Average power: 15.44 mW

Battery life: 30,780 J / 15.44 mW = 1,994,000 s = 554 hours = 23 days

This violates the six-month requirement. Continuous inference is too aggressive.

Conclusion:

Inference every 10 minutes (Option 1) or even every 1 minute (Option 2) satisfies the power budget with massive margin. The binding constraint is not inference frequency but communication frequency—LoRa transmission consumes 60 mJ per event, which dominates the energy budget if transmissions are frequent.

If you need to maximize battery life, reduce LoRa transmission frequency to once every 6 hours instead of once per hour. This reduces average power from 0.277 mW (1-minute inference, 1-hour LoRa) to:

Energy per 6-hour period:

Sensor: 7.5 mJ × 360 readings = 2,700 mJ

Inference: 7.92 mJ × 360 inferences = 2,851 mJ

LoRa: 60 mJ × 1 transmission = 60 mJ

Sleep: 3 V × 0.001 mA × (6 × 3600 - 360 × 5.22) s = 3 V × 0.001 mA × 19,720 s = 59.2 mJ

Total: 5,670 mJ per 6 hours = 21,600 s

Average power: 5,670 mJ / 21,600 s = 0.263 mW

Battery life: 30,780 J / 0.263 mW = 117 million s = 3.7 years

The system can run for nearly four years on two AA batteries with inference every minute and LoRa transmission every six hours. The power constraint is not the bottleneck—communication frequency is.

This is the power constraint workflow. You start with a target battery life, calculate the allowable average power, design an inference schedule, compute actual power consumption, and verify that the schedule meets the budget. If it doesn't, you reduce inference frequency, reduce transmission frequency, or optimize the model to reduce inference energy.

When You Cannot Meet the Power Budget

If the power budget cannot be satisfied with any reasonable inference schedule, you have four options:

Option 1: Use a larger battery or energy harvesting. This increases cost and size but solves the power constraint definitively.

Option 2: Reduce inference latency. Faster inference reduces active time, which reduces energy per inference. This often means upgrading to a faster processor or adding a hardware accelerator.

Option 3: Use event-driven triggering. Replace periodic inference with motion, threshold, or pattern-based triggers. This reduces inference frequency by 10–100× for sparse-event applications.

Option 4: Offload inference to the cloud. If communication cost (latency, energy, privacy) is acceptable, run inference on a server and transmit only sensor data. This trades inference energy for communication energy.

The power constraint is different from memory and compute constraints. Memory and compute are fixed by hardware—you cannot create more SRAM or more MACs/second without changing the chip. But power is variable—you can always reduce average power by reducing duty cycle, even if it means the system runs inference less frequently.

The question is whether the reduced inference frequency still meets the application requirement. If the answer is yes, the power constraint is solvable. If the answer is no, you've chosen the wrong hardware, the wrong model, or the wrong architecture for the application.

Chapter 8 examines one solution to simultaneous memory, compute, and power constraints: hardware accelerators that execute AI operations faster and more efficiently than general-purpose CPUs.


---

*[<- Chapter 6](./chapter-06-compute.md) | [Table of Contents](../README.md) | [Chapter 8 ->](./chapter-08-hardware-for-ai.md)*
