# Chapter 10: Real-Time AI: Reliability, Safety, and Deadlines

On March 18, 2018, an autonomous test vehicle operated by Uber struck and killed a pedestrian crossing the street in Tempe, Arizona. The vehicle's perception system—a combination of LIDAR, radar, and camera sensors feeding into neural network-based object detection—identified the pedestrian 5.6 seconds before impact. But the system's decision logic classified the detection as a false positive and suppressed the braking command. By the time the system reclassified the object as a pedestrian requiring emergency braking, 1.3 seconds remained—insufficient stopping distance at the vehicle's speed.
The root cause was not a failure of the neural network to detect the pedestrian. The network detected her. The failure was that the system lacked deterministic worst-case execution time guarantees for the perception-to-action pipeline. The neural network's inference time varied depending on scene complexity, and the system's decision logic could not bound the latency. When inference ran long, the action arrived too late.
This is the real-time AI problem. Neural networks are powerful but non-deterministic. Execution time depends on input data, cache state, memory contention, and scheduling. For soft real-time applications—smart thermostats, wearables, consumer electronics—variable latency is acceptable. Miss a deadline occasionally, and the user experience degrades slightly. But for hard real-time applications—automotive safety, industrial control, medical alarms—missing a deadline is a system failure that can cause physical harm.
This chapter teaches you to evaluate whether AI inference can be integrated into real-time systems without violating timing guarantees or safety requirements. By the end of this chapter, you will understand why determinism matters for real-time systems, identify the properties of neural networks that make worst-case execution time (WCET) analysis difficult, and apply design patterns that reduce the real-time risk of AI inference in safety-critical applications.
Hard vs. Soft Real-Time Requirements
A real-time system is one where correctness depends not just on the output but on when the output is produced. Real-time systems are classified by how they handle deadline violations.
Soft real-time: Deadlines are preferred but not mandatory. Missing a deadline degrades performance but does not cause failure. Video streaming is soft real-time—if a frame is delayed, the video stutters, but the system continues. A smart thermostat running gesture recognition is soft real-time—if inference takes 200 ms instead of 100 ms, the response feels sluggish, but the house doesn't burn down.
Firm real-time: Deadlines must be met for the result to be useful, but occasional misses are tolerated. A radar system processing aircraft returns is firm real-time—if processing completes after the next radar sweep, that data is useless, but missing one sweep doesn't crash the plane. Some fraction of deadline misses (e.g., <0.1%) is acceptable.
Hard real-time: Deadlines must be met with provable certainty. Missing a deadline is a system failure. Automotive airbag controllers are hard real-time—the deployment decision must occur within 10–20 ms of crash detection, or the airbag deploys too late to protect the occupant. Industrial safety systems that shut down machinery when faults are detected are hard real-time—late detection allows the fault to progress to catastrophic failure.
Neural network inference is inherently difficult to integrate into hard real-time systems because:
Execution time varies with input data.
Memory access patterns are irregular and cache-dependent.
Frameworks use dynamic memory allocation and complex control flow.
Worst-case execution time bounds are difficult to prove statically.
For soft real-time, these problems are manageable—you profile the model on realistic inputs, measure the 99th percentile latency, and add margin. For hard real-time, you must prove that even the worst-case input, under the worst-case cache state, with the worst-case scheduling, completes within the deadline. This is a much higher bar.
Non-Determinism in Neural Network Inference
Determinism means that executing the same code on the same input always takes the same amount of time. Non-determinism means execution time varies. Neural network inference has four sources of non-determinism:
1. Input-Dependent Execution Paths
Some neural network operations have execution time that depends on input values. Early-exit networks, for example, include branches that skip later layers if the model's confidence exceeds a threshold. An easy input (high confidence after three layers) takes 50 ms. A hard input (low confidence, must process all 10 layers) takes 200 ms.
Attention mechanisms, used in transformers and some RNNs, compute dynamic weights based on input content. The number of operations depends on which parts of the input are "attended to," which varies per input.
Even ReLU (rectified linear unit), a simple activation function, can introduce non-determinism if implemented with conditional branching:
for (int i = 0; i < n; i++) {
    output[i] = (input[i] > 0) ? input[i] : 0;
}

On processors with deep pipelines, the branch prediction behavior varies depending on the fraction of positive vs. negative inputs, which changes execution time by a few cycles per element.
2. Cache and Memory Effects
Modern processors use caches to speed up memory access. If data is in cache, access takes 1–2 cycles. If data is not in cache (a cache miss), access takes 10–100 cycles while the data is fetched from main memory.
Whether a memory access hits or misses in cache depends on:
What data was accessed recently (cache is warm or cold)
Whether other tasks or interrupts evicted data from the cache
Memory access patterns (sequential accesses hit more often than random accesses)
For neural network inference, cache behavior is input-dependent. A convolutional layer processing a smooth image (low spatial variation) might have better cache locality than a noisy image (high spatial variation), leading to 10–30% variation in execution time.
3. RTOS Scheduling and Preemption
On systems running an RTOS, inference may be preempted by higher-priority tasks. If inference runs at normal priority and a communication event arrives, the RTOS suspends inference, handles the event, and resumes inference. The total wall-clock time is inference execution time plus preemption time.
Preemption time is non-deterministic because it depends on when interrupts arrive, how long interrupt handlers run, and whether higher-priority tasks are ready to execute. Bounding preemption time requires analyzing all tasks in the system, not just the inference task.
4. Framework Overhead
Inference frameworks (TensorFlow Lite Micro, PyTorch Mobile, ONNX Runtime) add abstraction layers—graph interpreters, dynamic dispatch, memory managers. These introduce variability:
Dynamic memory allocation (malloc/free) has variable execution time depending on heap fragmentation.
Virtual function calls and dynamic dispatch have variable latency depending on instruction cache state.
Framework profiling or logging code (if enabled) adds overhead.
For safety-critical applications, frameworks with dynamic allocation and complex control flow are unsuitable. The inference engine must be statically analyzable—no dynamic allocation, no recursion, no unbounded loops.
WCET Analysis for AI: What Works, What Doesn't, and Open Problems
Worst-case execution time (WCET) analysis aims to compute the maximum time a piece of code can take to execute, under any input and any system state, without actually running it on all possible inputs (which is infeasible).
Traditional WCET analysis tools (AbsInt aiT, Rapita RVS, SWEET) work by:
Modeling the processor's pipeline, cache, and memory system.
Constructing a control-flow graph of the code.
Finding the longest execution path through the graph, accounting for pipeline stalls and cache misses.
Returning an upper bound on execution time.
This works well for simple embedded code—sensor drivers, control loops, state machines—where control flow is explicit and data-independent. It fails for neural network inference because:
Path explosion: A network with 20 layers and 10 possible execution paths per layer has 10²⁰ possible paths. Enumerating all paths is computationally infeasible.
Indirect control flow: Frameworks use function pointers and dynamic dispatch. Static analysis cannot determine which function will execute without runtime information.
Complex cache behavior: Convolutional layers access memory in patterns that depend on input dimensions, stride, padding—parameters that vary per layer. Modeling cache behavior for all possible layer configurations is intractable.
Floating-point variability: IEEE 754 floating-point arithmetic is not associative. Reordering operations (which compilers do for optimization) can change results slightly, and some processors have variable-latency floating-point units depending on the operand values (denormals, NaNs).
For neural networks, static WCET analysis is an open research problem. Current tools cannot provide tight, provable bounds for general neural networks on general-purpose processors.
The pragmatic workarounds are:
Empirical WCET measurement: Run the model on thousands of inputs, measure execution time for each, and take the maximum observed time plus a safety margin (20–50%). This is not a proof, but it's practical.
Restrict the network architecture: Use only layers with data-independent execution paths. Ban early exits, dynamic attention, conditional branches. This makes static analysis feasible but limits model expressiveness.
Disable caches or use cache locking: Lock critical data (weights, activations) into cache or scratchpad memory so all memory accesses are deterministic. This sacrifices performance but makes timing predictable.
Run inference at highest priority with interrupts disabled: Eliminate preemption and interrupt latency as sources of variability. This increases latency for other tasks but makes inference timing predictable.
Use hardware accelerators with provable timing: Some NPUs (e.g., automotive-grade neural accelerators from NXP, Renesas) provide WCET guarantees for fixed-point models. The accelerator's timing is deterministic because it has no caches and no dynamic control flow.
None of these is a complete solution. Each trades off expressiveness, performance, or cost to gain determinism.
Safety-Critical AI: Functional Safety Standards and What They Require
Safety-critical systems—automotive, medical, industrial—are governed by functional safety standards that specify how systems must be designed, tested, and verified to ensure they do not cause harm.
Key standards:
IEC 61508: Generic functional safety standard for electrical/electronic systems. Defines Safety Integrity Levels (SIL 1–4), where SIL 4 is the most stringent.
ISO 26262: Automotive-specific adaptation of IEC 61508. Defines Automotive Safety Integrity Levels (ASIL A–D), where ASIL D is the most stringent.
IEC 62304: Medical device software lifecycle standard.
DO-178C: Avionics software standard.
These standards require:
Hazard analysis: Identify all possible failure modes and their consequences.
Risk classification: Classify failures by severity (catastrophic, hazardous, major, minor).
Safety requirements: Specify maximum tolerable failure rates (e.g., <10⁻⁹ failures per hour for ASIL D).
Verification and validation: Prove that the system meets safety requirements through testing, formal methods, or certification.
Neural networks pose unique challenges for functional safety:
Blackbox behavior: Neural networks are not programmed with explicit rules. Their behavior emerges from learned weights, which are opaque. You cannot inspect a neural network and determine "this will always detect pedestrians within 5.6 seconds" the way you can inspect a threshold-based detector.
No formal specification: Traditional software has a specification (input → expected output). Neural networks have training data and statistical performance metrics (95% accuracy on validation set). But statistical performance is not a safety guarantee. A network with 95% accuracy fails 5% of the time—is that acceptable for a safety-critical application? Depends on the failure mode. Missing a pedestrian is catastrophic. Misclassifying a stop sign as a speed limit sign is hazardous.
Dataset dependency: A network's behavior is only as good as its training data. If the training set doesn't include edge cases (pedestrians wearing reflective vests at night, stop signs partially occluded by snow), the network will fail on those cases in deployment. Proving completeness of the training set is impossible.
Adversarial vulnerability: Neural networks can be fooled by adversarially crafted inputs—images or signals designed to cause misclassification. A stop sign with specific stickers can be misclassified as a speed limit sign. For safety-critical systems, adversarial robustness is a requirement, not an optional property.
Functional safety standards are being extended to address AI:
ISO/PAS 21448 (SOTIF): Safety of the intended functionality—addresses the case where a system behaves as designed but the design is insufficient (e.g., a pedestrian detector that was never trained on wheelchairs).
UL 4600: Standard for autonomous vehicles, includes requirements for AI validation.
ISO/TR 5469: Guidance on AI in road vehicles.
But these standards are not yet mature. Certifying an AI component for ASIL D (the highest automotive safety level) is still a research problem. The current practice is to use AI in non-safety-critical roles (driver assistance, predictive maintenance diagnostics) and keep safety-critical decisions in traditional, certifiable software.
Design Patterns for Real-Time-Safe AI
If you must integrate AI into a real-time or safety-critical system, several design patterns reduce risk.
Pattern 1: AI as Advisory, Not Command
The AI model provides a recommendation, but a deterministic supervisor makes the final decision. The supervisor has hard real-time guarantees and a proven safety envelope. The AI recommendation is treated as one input among many.
Example: An industrial robot uses AI for defect detection but also has a hardware limit switch. If the AI detects a defect, it flags the part for manual inspection. If the part physically hits the limit switch (AI failed to detect a defect), the robot stops immediately.
The AI improves performance (fewer false positives, better classification), but the safety fallback is independent of AI.
Pattern 2: Bounded Execution with Rejection
The AI model runs with a hard execution time budget. If execution exceeds the budget, the inference is aborted, and the system falls back to a safe default action.
Implementation:
Start a hardware timer when inference begins.
If the timer expires before inference completes, terminate inference and return "unknown."
The application handles "unknown" as a safe state (e.g., brake, sound alarm, request human intervention).
This guarantees that the AI path never violates the deadline, but it introduces a new failure mode: rejected inferences. The system must be designed so that rejections are safe.
Pattern 3: Confidence Gating
The AI model outputs not just a classification but a confidence score. Only high-confidence predictions are acted upon. Low-confidence predictions are treated as "unknown" and trigger a fallback.
Example: A pedestrian detector outputs "pedestrian detected" with 98% confidence → trigger braking. The same detector outputs "pedestrian detected" with 60% confidence → don't trigger braking, but flag for human review.
This reduces the probability of catastrophic false positives (acting on a misclassified object) but increases false negatives (failing to act on a real pedestrian with low-confidence detection).
The confidence threshold is a safety parameter that must be tuned based on hazard analysis. For life-critical applications, the threshold is set conservatively high (e.g., 99.9% confidence required), which means the AI rarely acts, but when it does, it's highly reliable.
Pattern 4: Redundant Inference with Voting
Run multiple models in parallel—different architectures, different training sets—and use majority voting or consensus to make decisions. If two out of three models agree, act on the consensus. If all three disagree, reject the inference.
This improves robustness to adversarial inputs and model-specific failure modes. But it triples inference cost (latency, power, memory), so it's only viable when resources allow.
Pattern 5: Hierarchical Safety Layers
Layer safety mechanisms so that AI failure is caught by a subsequent, simpler, provably safe check.
Example: An autonomous vehicle's perception stack:
Layer 1 (AI): Neural network for pedestrian detection. High accuracy, complex, non-deterministic.
Layer 2 (Rule-based): Simple threshold detector on LIDAR distance. Low accuracy, but deterministic and proven safe. If Layer 1 fails to detect a pedestrian, Layer 2 still triggers braking if any object is within 5 meters.
Layer 3 (Hardware): Physical bumper sensor. If Layers 1 and 2 fail, the bumper triggers an emergency stop.
Each layer is progressively simpler and more conservative. The AI optimizes for performance, but the safety guarantee comes from the fallback layers.
Failure Mode Analysis for AI in Embedded Systems
Functional safety standards require identifying all failure modes and their consequences. For AI components, failure modes include:
False negatives (misses): The model fails to detect a condition that is present. A pedestrian detector misses a pedestrian → no braking → collision. Consequence: catastrophic.
False positives (false alarms): The model detects a condition that is not present. A fault detector flags a healthy motor as faulty → unnecessary shutdown → production loss. Consequence: economic but not safety-critical.
Latency violations: The model produces the correct output but too late. A collision detector identifies an obstacle at 5 meters but inference takes 500 ms, by which time the vehicle has moved to 2 meters → insufficient braking distance. Consequence: catastrophic.
Adversarial misclassification: The model is fooled by a deliberately crafted input. A stop sign with stickers is classified as a speed limit sign → vehicle fails to stop. Consequence: catastrophic.
Data distribution shift: The model encounters inputs outside its training distribution. A pedestrian detector trained only on adults fails on children or wheelchair users → missed detection. Consequence: catastrophic.
Numerical instability: Quantization or fixed-point arithmetic introduces errors that accumulate. A regression model predicts -5% where the true value is +5% → incorrect decision. Consequence: depends on application.
For each failure mode, you must:
Estimate the probability (based on validation testing, field data, or theoretical analysis).
Classify the severity (catastrophic, hazardous, major, minor).
Determine whether the probability × severity is within the acceptable risk threshold.
If not, apply mitigations (design patterns, redundancy, human oversight).
For safety-critical systems, the acceptable failure rate is extremely low. ASIL D (automotive) requires <10⁻⁸ failures per hour of operation. If your AI model has a 1% false negative rate, it fails 10,000× more often than allowed. You must either improve the model to 99.999999% accuracy (impossible for most tasks) or add redundancy and fallbacks so that AI failures are caught before they cause harm.
A Worked Example: Integrating AI into a Hard Real-Time Industrial Control Loop
You are integrating an anomaly detector into a hard real-time control loop for a high-speed CNC milling machine. The machine operates at 10,000 RPM. The control loop runs at 1 kHz (every 1 ms). If an anomaly is detected, the machine must stop within 5 ms to prevent tool breakage or workpiece damage.
You have a neural network trained to detect bearing faults from vibration data:
Input: 128-sample window of 3-axis accelerometer data (384 values)
Architecture: 1D CNN with 3 convolutional layers + 2 fully connected layers
Parameters: 80,000 (int8 quantized, 80 KB)
Operation count: 12 million MACs
Measured latency: 35–55 ms (on target hardware, Cortex-M7 at 400 MHz)
Problem: The inference latency (35–55 ms) exceeds the 5 ms deadline by an order of magnitude. Direct integration is impossible.
Solution: Apply Design Patterns
Step 1: Bounded Execution with Rejection
Set a 5 ms hard timeout. If inference doesn't complete in 5 ms, abort and return "unknown." But this doesn't help—inference takes 35 ms minimum, so every inference is aborted. The AI provides no value.
Step 2: AI as Advisory, Not Command
Run inference asynchronously at a slower rate (e.g., every 50 ms, not every 1 ms). The AI provides fault predictions that update a fault probability estimate. The hard real-time control loop checks the fault probability every 1 ms. If the probability exceeds a threshold, trigger shutdown.
Implementation:
Control loop: Runs every 1 ms. Reads fault_probability variable. If fault_probability > 0.95, trigger emergency stop.
AI task: Runs every 50 ms. Executes inference (35–55 ms), updates fault_probability based on the result.
Default: fault_probability = 0.0 (no fault detected).
Timing analysis:
Control loop deadline: 1 ms. Loop execution time (reading a variable and comparing to a threshold): <10 µs. Meets deadline with 100× margin.
AI task deadline: None (advisory role). Inference can take 55 ms without violating any hard deadline.
Safety analysis:
If AI fails to detect a fault, fault_probability stays low, and the machine continues running. This is a false negative, but the consequence is deferred (the fault progresses until detected by other means or until failure). Not ideal, but not immediately catastrophic.
If AI falsely detects a fault, fault_probability goes high, and the machine stops. This is a false positive, causing production loss but not safety harm.
Failure mode: AI never updates fault_probability (inference crashes or hangs). Fallback: Add a watchdog timer that resets fault_probability to 0.0 if the AI task doesn't update it within 100 ms. This ensures that a stuck AI task doesn't cause a spurious shutdown.
Step 3: Hierarchical Safety Layers
Add a simple, deterministic backup fault detector that runs in the control loop.
Layer 1 (AI): Neural network running every 50 ms, updates fault_probability.
Layer 2 (Rule-based): RMS vibration threshold detector running every 1 ms. If RMS exceeds a conservative threshold, set fault_flag = true immediately.
Layer 3 (Hardware): Current sensor on the motor. If current exceeds safe limits (indicating mechanical seizure), hardware trips the power supply.
Control loop logic:
if (fault_flag || fault_probability > 0.95) {
    trigger_emergency_stop();
}

Now the system has defense in depth:
AI (Layer 1) provides early warning of incipient faults with 35–55 ms latency.
Rule-based detector (Layer 2) provides conservative, deterministic fault detection with <1 ms latency.
Hardware protection (Layer 3) is the ultimate safety net if both software layers fail.
The AI improves system performance (earlier fault detection, fewer false positives than the rule-based detector) without being in the critical path for safety.
Result:
The AI is successfully integrated into a hard real-time system by removing it from the hard real-time path. The control loop meets its 1 ms deadline. The AI provides value by improving fault detection accuracy. Safety is guaranteed by deterministic fallbacks that do not depend on AI.
This is the pattern for integrating non-deterministic AI into deterministic real-time systems: isolate the AI, make it advisory, and layer safety mechanisms so that AI failure is tolerable.
When AI Cannot Be Integrated into Hard Real-Time Systems
Some applications cannot tolerate the non-determinism of neural networks, and no design pattern fixes it. These applications require provable, certifiable determinism:
Nuclear reactor control: Failure consequences are catastrophic and irreversible. Control logic must be formally verified, which neural networks cannot be.
Aircraft flight control: Hard real-time deadlines (milliseconds), provable WCET required for certification (DO-178C), and failure modes are unacceptable. AI is used in avionics for optimization and monitoring but not for primary flight control.
Medical device critical functions: Pacemakers, defibrillators, insulin pumps—primary control logic must be certifiable under IEC 62304. AI is used for diagnostics and trend analysis but not for life-critical actuation.
For these applications, AI is limited to non-critical roles:
Diagnostics and monitoring: AI flags anomalies for human review but doesn't take automated action.
Optimization: AI tunes parameters for efficiency but operates within safety bounds enforced by deterministic logic.
Post-processing: AI analyzes logged data after the fact to improve future designs but doesn't affect real-time operation.
The hard boundary is this: if you cannot prove that the AI component meets WCET and reliability requirements for the applicable safety standard, you cannot deploy it in a safety-critical role. Use AI where it adds value without adding risk.
What Comes Next
You have now completed Part II—Inference-First Evaluation. You can audit any proposed AI deployment against all major constraint dimensions: memory (Chapter 5), compute (Chapter 6), power (Chapter 7), hardware acceleration (Chapter 8), communication (Chapter 9), and real-time guarantees (Chapter 10).
You can evaluate whether a model fits, whether it runs fast enough, whether it drains the battery, whether it needs an accelerator, whether it should offload to the cloud, and whether it can meet hard deadlines. You can identify every constraint failure before deployment.
But all of this assumes the model is given. You evaluate feasibility, not optimality. Part III—Model-Aware Design—extends the toolkit into model selection and optimization. The model is no longer fixed. It becomes a design variable. Chapter 11 teaches you to select models for deployment constraints, not just for accuracy benchmarks.

---

*[← Chapter 9](./chapter-09-communication-edge-cloud.md) | [Table of Contents](../README.md) | [Chapter 11 →](./chapter-11-model-selection.md)*