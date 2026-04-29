# Chapter 14: Integration Case Studies: Full Design Decisions

You have built the complete toolkit. You can quantify embedded constraints (Chapters 2, 5–7), evaluate hardware acceleration options (Chapter 8), design edge-cloud architectures (Chapter 9), assess real-time safety requirements (Chapter 10), select models for deployment (Chapter 11), optimize models through quantization and pruning (Chapter 12), and navigate deployment toolchains (Chapter 13).

Each skill was taught in isolation. This chapter integrates them. You will apply the full decision framework—from application requirements to deployed system—on three realistic embedded AI problems. Each case study presents a complete design decision workflow: requirements analysis, constraint quantification, hardware selection, model selection, optimization, deployment strategy, and the alternatives that were rejected.

The case studies are:

Case A: Industrial vibration monitoring for predictive maintenance

Case B: Medical wearable for cardiac arrhythmia detection

Case C: Agricultural edge sensor for crop disease detection

By the end of this chapter, you will have seen the complete integration process three times, from three different angles. You will understand how to produce a structured integration decision document that specifies the model, hardware, deployment strategy, and constraint justification. And you will be able to critique proposed AI embedded designs and identify the decisions most likely to fail in production.

This is the terminal capability the book was built to deliver.

Case A: Industrial Anomaly Detection — Vibration Monitoring for Predictive Maintenance

Application Requirements

A manufacturing facility operates 200 CNC milling machines that run 24/7. Each machine has critical bearings that degrade over time. When a bearing fails catastrophically, it damages the machine ($50,000 repair cost) and halts production ($10,000 per hour of downtime). The facility wants to deploy vibration sensors with on-device anomaly detection to predict bearing failures 24–48 hours in advance, allowing scheduled maintenance instead of emergency repairs.

Functional Requirements:

Monitor vibration continuously at 1 kHz sampling rate (3-axis accelerometer)

Detect bearing faults with ≥95% sensitivity (true positive rate)

False positive rate <5% (no more than 1 false alarm per machine per month)

Alert latency: detection within 10 seconds of fault onset is acceptable (soft real-time)

Physical Constraints:

Battery-powered (18-month target life, coin cell or small Li-ion)

No cloud connectivity (factory has no WiFi, cellular is cost-prohibitive at $5/month/device × 200 devices)

Harsh environment (vibration, heat, dust—industrial-grade hardware required)

Cost: <$40 per sensor (total deployment budget: $8,000)

Safety Requirements:

Not a safety-critical system (failure to detect a fault is costly but not dangerous)

No hard real-time requirements (alerts are for scheduling, not for emergency shutdown)

Constraint Quantification

Power budget:

 18-month life on battery. Assume a 1000 mAh Li-ion cell at 3.7V (13,320 J total). Target average power:

P_avg = 13,320 J / (18 months × 730 hours/month × 3600 s/hour) = 0.28 mW

This is aggressive. Even sleep current of 10 µA (3.7V × 0.01 mA = 0.037 mW) consumes 13% of the budget. Inference must be extremely infrequent or very fast.

## Latency budget:

 Detection within 10 seconds is acceptable. This is soft real-time—occasional delays are tolerable.

Memory budget:

 Not specified. Assume typical low-cost microcontroller: 256 KB SRAM, 1 MB flash.

Connectivity:

 No cloud. Must store alerts locally and transmit via local gateway (Modbus, RS-485) once per day or on-demand.

Decision 1: Processing Architecture (On-Device vs. Edge Gateway)

Option 1: Fully on-device processing.

 Sensor runs inference locally, stores alerts in flash, and transmits summary once per day to a local gateway.

Pros: Minimal communication cost (one transmission per day). Privacy-preserving (raw vibration data never leaves the sensor).

Cons: Must fit model on microcontroller. Power budget is extremely tight.

Option 2: Edge gateway processing.

 Sensor streams raw vibration data to a local industrial PC (edge gateway) via wired connection (RS-485 or CAN bus). Gateway runs inference.

Pros: Gateway is mains-powered—no energy constraint. Can run larger, more accurate models.

Cons: Requires wiring 200 sensors to gateways (installation cost). Gateway is a single point of failure.

Option 3: Hybrid—on-device triggers, gateway refinement.

 Sensor runs a simple threshold detector locally. When vibration exceeds threshold, sensor wakes a radio and transmits a 1-second window to the gateway for ML-based classification.

Pros: Low power (radio is off 99% of time). Gateway handles complex ML.

Cons: Requires wireless (LoRa or sub-GHz radio), which adds cost and complexity.

Decision: Select Option 1 (fully on-device). Battery life is tight, but communication infrastructure (Option 2) or wireless radios (Option 3) increase cost beyond the $40/unit budget. The challenge is achieving 18-month battery life with on-device inference.

## Decision 2: Model Selection

Bearing fault detection is a time-series anomaly detection problem. Candidate approaches:

Approach A: Threshold detector (classical).

 Compare RMS vibration to a learned threshold. Flag anomaly if RMS exceeds threshold for >5 consecutive seconds.

Pros: Minimal compute (RMS calculation is cheap). Deterministic.

Cons: Low accuracy (60–70% sensitivity in prior studies). High false positive rate (normal transient events trigger alarms).

Approach B: Decision tree on engineered features.

 Extract 20 features from 1-second windows (mean, variance, spectral peaks, kurtosis, zero-crossing rate). Train random forest (50 trees).

Pros: Low memory (200 KB model). Low latency (5 ms inference). High accuracy (85–90% sensitivity in literature).

Cons: Requires hand-engineered features. May not generalize to different machine types.

Approach C: 1D CNN on raw vibration.

 Train a 1D convolutional network on 1-second windows of raw 3-axis vibration data (3,000 samples).

Pros: No feature engineering. Can learn subtle patterns that hand-crafted features miss. Accuracy 92–96% in recent research.

Cons: Higher memory (500 KB – 1 MB models typical). Higher latency (50–200 ms). Higher power.

Approach D: Autoencoder (unsupervised).

 Train an autoencoder on normal vibration. Detect anomalies as high reconstruction error.

Pros: Does not require labeled fault data (normal-only training). Can detect novel faults.

Cons: High memory (autoencoders need encoder + decoder, often 1–2 MB). Difficult to set reconstruction threshold (false positives vs. false negatives tradeoff).

Decision: Select Approach B (random forest on features). Rationale:

Approach A fails the accuracy requirement (70% < 95% sensitivity).

Approach C is too expensive in memory and power for 18-month battery life.

Approach D requires more complex threshold tuning and is hard to validate.

Approach B achieves 88% sensitivity (below 95% target, but can be improved with better features), fits in memory, and runs in 5 ms—enabling low power through infrequent inference.

Decision 3: Hardware Selection

Constraints:

Flash: 200 KB model + 600 KB firmware = 800 KB required

SRAM: 50 KB (feature extraction buffers) + 20 KB (firmware) = 70 KB required

Power: Extreme—0.28 mW average

Cost: <$40

Candidates:

Option 1: STM32L4R5 (ultra-low-power MCU)

Cortex-M4 at 120 MHz, 640 KB SRAM, 2 MB flash

Sleep current: 0.4 µA, Active: 100 µA/MHz = 12 mA at 120 MHz

Cost: $6

Option 2: nRF52840 (Bluetooth Low Energy MCU)

Cortex-M4 at 64 MHz, 256 KB SRAM, 1 MB flash

Sleep current: 2.8 µA, Active: 5.3 mA at 64 MHz

Cost: $4 (but includes Bluetooth radio, which is not needed)

Option 3: ESP32-S3 (WiFi/BLE SoC)

Dual-core Xtensa at 240 MHz, 512 KB SRAM, 8 MB flash

Sleep current: 10 µA, Active: 40 mA

Cost: $4.50

Decision: Select STM32L4R5. Rationale:

Memory: 640 KB SRAM exceeds requirement by 9×. 2 MB flash exceeds requirement by 2.5×. Comfortable margin.

Power: 0.4 µA sleep current is the lowest among options. This is critical for 18-month battery life.

Cost: $6 is within budget ($40 - $6 leaves $34 for accelerometer, enclosure, battery, PCB).

nRF52840 is cheaper but has less SRAM and higher sleep current (2.8 µA vs. 0.4 µA—7× worse). ESP32-S3 has even higher sleep current (10 µA—25× worse).

Decision 4: Inference Scheduling Strategy

Given P_avg target of 0.28 mW, calculate sustainable inference frequency.

Energy per inference:

Feature extraction: 10 ms at 12 mA = 3.7V × 0.012 A × 0.01 s = 0.44 mJ

Random forest inference: 5 ms at 12 mA = 3.7V × 0.012 A × 0.005 s = 0.22 mJ

Total per inference: 0.66 mJ

Sleep power:

3.7V × 0.0004 mA = 0.00148 mW

Inference interval: If inference runs every T seconds, average power is:

P_avg = (0.66 mJ / T) + 0.00148 mW

Set P_avg = 0.28 mW:

0.28 mW = (0.66 mJ / T) + 0.00148 mW

 0.2785 mW = 0.66 mJ / T

 T = 0.66 mJ / 0.0002785 W = 2.37 seconds

Inference can run every 2.4 seconds while staying within power budget.

Validation:

Faults develop over hours to days. Checking every 2.4 seconds provides 1,500 checks per hour, which is more than sufficient to detect gradual degradation.

Latency requirement (10 seconds) is met: worst-case detection time is one inference interval (2.4 s).

Decision: Run inference every 2.5 seconds.

Decision 5: Model Training and Validation

Train random forest on:

Normal data: 10,000 samples (vibration from healthy machines)

Fault data: 2,000 samples (vibration from machines with known bearing faults)

Features extracted per 1-second window:

RMS (3 axes) = 3 features

Peak amplitude (3 axes) = 3 features

Spectral centroid (FFT-based, 3 axes) = 3 features

Kurtosis (3 axes) = 3 features

Zero-crossing rate (3 axes) = 3 features

Total: 15 features per sample.

Train random forest with 50 trees, max depth 10.

Validation results:

Sensitivity (true positive rate): 89%

False positive rate: 3%

Model size: 180 KB

Sensitivity is below the 95% target. To improve, add more features (dominant FFT frequency, harmonic ratios) and retrain.

After feature engineering iteration 2:

Features: 25 (added 10 frequency-domain features)

Sensitivity: 94%

False positive rate: 4%

Still below 95%, but close. Product team accepts 94% sensitivity as acceptable given cost constraints.

Decision 6: Deployment

Firmware stack:

STM32 HAL drivers: 200 KB flash, 10 KB SRAM

Random forest inference engine (custom, lightweight): 20 KB flash

Feature extraction library (FFT, statistical features): 100 KB flash

Application logic (data acquisition, scheduling, storage): 50 KB flash

Model (random forest weights): 180 KB flash

Total flash: 550 KB (fits in 2 MB with 73% margin)

Total SRAM: 50 KB feature buffers + 10 KB firmware = 60 KB (fits in 640 KB with 91% margin)

Flash firmware, deploy to 5 pilot units, and run field trial for 1 month.

Field Trial Results

Battery life: 16.5 months (measured extrapolation from 1-month current draw)

Sensitivity: 91% (3 faults detected out of 3.3 expected based on historical failure rates—small sample)

False positives: 2 per unit per month (8% false positive rate, higher than validation)

Root cause of higher false positives: Vibration from adjacent machines caused false triggers. Fixed by adding spatial filtering (compare to adjacent sensor readings—if only one sensor triggers, it's likely a false positive).

After firmware update, false positive rate: 4.5%.

Final System Specification

Hardware: STM32L4R5, 3-axis MEMS accelerometer (ADXL355), 1000 mAh Li-ion battery

 Model: Random forest (50 trees, 25 features, 180 KB)

 Inference schedule: Every 2.5 seconds

 Performance: 91% sensitivity, 4.5% false positive rate, 16.5-month battery life

 Cost: $38 per unit (hardware BOM)

Alternatives rejected and why:

1D CNN: Memory and power constraints. Would require larger battery (higher cost) or more frequent charging (unacceptable for industrial deployment).

Cloud processing: No connectivity infrastructure. Adding cellular would cost $5/month/unit × 200 = $1,000/month ongoing.

Edge gateway architecture: Installation cost of wiring 200 sensors exceeded budget.

Case B: Medical Wearable — Cardiac Arrhythmia Detection

Application Requirements

A medical device company is developing a wearable ECG monitor for continuous cardiac rhythm monitoring. The device detects arrhythmias (irregular heartbeats) and alerts the user to seek medical attention. The device is intended for outpatient monitoring of patients with known cardiac conditions.

Functional Requirements:

Continuous ECG monitoring at 250 Hz sampling rate (single-lead ECG)

Detect 5 arrhythmia types: atrial fibrillation, ventricular tachycardia, premature ventricular contractions, bradycardia, tachycardia

Sensitivity ≥98% for life-threatening arrhythmias (VT, VF)

Specificity ≥95% (false positive rate <5%)

Latency: Alert within 30 seconds of arrhythmia onset

Regulatory Requirements:

FDA Class II medical device (510(k) pathway)

Must demonstrate clinical validation (sensitivity/specificity on diverse patient population)

Software lifecycle per IEC 62304

Cybersecurity requirements (if device transmits data)

Physical Constraints:

Wearable form factor (wristband or chest patch)

7-day battery life minimum (rechargeable Li-ion)

Lightweight (<50 grams)

Cost: <$200 per unit (consumer medical device pricing)

Constraint Quantification

Power budget:

 7 days on 500 mAh Li-ion cell at 3.7V (6,660 J). Average power:

P_avg = 6,660 J / (7 days × 24 hours × 3600 s) = 11 mW

Latency budget:

 30 seconds for alert. This allows batched processing (accumulate 10–30 seconds of ECG data, run inference on batch).

Memory budget:

 Not specified. Assume wearable SoC: 512 KB SRAM, 2 MB flash.

Privacy:

 Medical data (ECG) is HIPAA-protected. Cannot send raw ECG to cloud without encryption and patient consent. On-device processing preferred.

Decision 1: Processing Architecture

Option 1: Fully on-device.

 Run arrhythmia detection on the wearable. Send only alerts (not raw ECG) to a paired smartphone.

Pros: HIPAA-compliant (raw data never leaves device). Works offline (no connectivity required).

Cons: Must fit ML model on wearable SoC. Power budget is tight.

Option 2: Smartphone processing.

 Stream raw ECG to smartphone via Bluetooth. Smartphone runs inference and displays alerts.

Pros: Smartphone has more compute and power. Can run larger, more accurate models.

Cons: Requires Bluetooth active continuously (high power—20 mA continuous). HIPAA risk (raw ECG on smartphone is harder to secure than on medical-grade wearable).

Option 3: Hybrid—on-device screening, smartphone confirmation.

 Wearable runs a fast screening model. When potential arrhythmia is detected, wearable sends 10-second ECG window to smartphone for confirmation by a larger model.

Pros: Low power (Bluetooth only active during arrhythmia events, which are rare). High accuracy (smartphone model is more accurate).

Cons: Complex firmware (two models, handoff logic). Latency depends on Bluetooth connection.

## Decision: Select Option 1 (fully on-device). Rationale:

Regulatory: FDA prefers on-device processing for medical algorithms (deterministic, no connectivity dependencies).

Privacy: HIPAA compliance is simpler when raw ECG never leaves the device.

Reliability: Device must work even if smartphone is out of range or battery is dead.

The power budget (11 mW) is tight but feasible with duty-cycled inference.

## Decision 2: Model Selection

Arrhythmia detection from ECG is a time-series classification problem. Candidates:

Approach A: Rule-based detector (classical).

 QRS complex detection + RR interval analysis + morphology rules. This is how traditional Holter monitors work.

Pros: Well-validated. Deterministic. Low power.

Cons: Accuracy is 85–92% (literature), below 98% requirement. Struggles with noisy ECG or unusual morphologies.

Approach B: 1D CNN on raw ECG.

 Train CNN on 10-second ECG windows (2,500 samples). Output: normal vs. 5 arrhythmia classes.

Pros: State-of-the-art accuracy (98%+ for VT/VF in recent papers). Learns features from raw signal.

Cons: Large models (1–5 MB typical for high accuracy). High compute (50–200 ms inference).

Approach C: LSTM on RR intervals.

 Extract RR intervals (time between heartbeats) using a simple peak detector. Feed RR sequence to LSTM. Output: arrhythmia class.

Pros: Smaller input (RR intervals, not raw ECG). Lower memory (100–500 KB models).

Cons: LSTM state memory (64–128 units × 4 bytes = 256–512 bytes per time step, accumulates). Accuracy depends on QRS detection quality.

Approach D: Hybrid—CNN for feature extraction + lightweight classifier.

 Use a small CNN to extract features from 10-second ECG windows. Feed features to a decision tree or small fully-connected network.

Pros: Smaller than full CNN. Faster inference. Still achieves 95–97% accuracy.

Cons: Two-stage complexity. Accuracy slightly below full CNN.

Decision: Select Approach B (1D CNN on raw ECG), with aggressive optimization. Rationale:

Regulatory: FDA expects high sensitivity (≥98%) for life-threatening arrhythmias. Only Approach B reliably achieves this in literature.

Robustness: CNN on raw ECG is more robust to QRS detection errors than RR-interval-based approaches.

Risk tolerance: Medical device—accuracy is paramount, power is secondary (users can recharge nightly).

Challenge: Compress a state-of-the-art CNN (typically 3–5 MB) to fit on a wearable SoC (2 MB flash, 512 KB SRAM).

## Decision 3: Model Optimization

Start with a well-validated architecture from literature: ResNet-inspired 1D CNN with 18 layers, 2.5 million parameters (10 MB float32).

Optimization sequence:

Step 1: Structured pruning.

 Remove 40% of channels across all layers. Retrain to recover accuracy.

 Result: 1.5 million parameters (6 MB float32), 97.2% sensitivity for VT/VF (below 98% target).

Step 2: Knowledge distillation.

 Train a smaller student network (8 layers, 800,000 parameters) to mimic the pruned model's behavior.

 Result: 800,000 parameters (3.2 MB float32), 98.1% sensitivity. Memory still too large.

Step 3: Quantization (int8 QAT).

 Apply quantization-aware training to the distilled model.

 Result: 800 KB (int8), 97.8% sensitivity (0.3% accuracy loss due to quantization).

Sensitivity for VT/VF is 97.8%, just below the 98% threshold. Product team accepts this with a plan to improve through post-market data collection and model updates.

Final model: 800 KB flash, 180 KB SRAM (activations).

Decision 4: Hardware Selection

Constraints:

Flash: 800 KB model + 600 KB firmware = 1.4 MB

SRAM: 180 KB activations + 100 KB firmware = 280 KB

Power: 11 mW average

Cost: <$200 (device cost allows $30–50 for SoC)

Candidates:

Option 1: Nordic nRF5340 (BLE SoC)

Dual-core (Cortex-M33 at 128 MHz + Cortex-M33 at 64 MHz)

1 MB flash, 512 KB SRAM

BLE 5.2, low-power design

Cost: $8

Option 2: STM32L4R5 (ultra-low-power MCU)

Cortex-M4 at 120 MHz

2 MB flash, 640 KB SRAM

No integrated radio (requires external BLE module)

Cost: $6 + $5 (BLE module) = $11

Option 3: STM32U5 (ultra-low-power with TrustZone)

Cortex-M33 at 160 MHz

2 MB flash, 786 KB SRAM

TrustZone for secure boot and firmware updates (important for medical devices)

Cost: $12

Decision: Select STM32U5. Rationale:

Memory: 2 MB flash and 786 KB SRAM comfortably exceed requirements.

Security: TrustZone enables secure boot and encrypted firmware updates, which FDA expects for connected medical devices.

Power: Ultra-low-power design with multiple sleep modes.

Cost: $12 is well within budget.

nRF5340 is cheaper and includes BLE, but lacks TrustZone. For a medical device, security features justify the cost premium.

## Decision 5: Inference Scheduling

ECG sampled at 250 Hz. Arrhythmia detection operates on 10-second windows (2,500 samples). Inference latency on STM32U5 (measured): 120 ms.

Strategy: Batch processing. Accumulate 10 seconds of ECG, run inference, sleep. Repeat.

Duty cycle: 120 ms active every 10 seconds = 1.2% duty cycle.

Power calculation:

Active (inference): 3.7V × 25 mA × 0.12 s every 10 s = 11.1 mJ per cycle

ECG acquisition (continuous): 3.7V × 1 mA × 10 s = 37 mJ per cycle

Sleep (processor): 3.7V × 0.005 mA × 9.88 s = 0.18 mJ per cycle

Total per cycle: 48.5 mJ

Average power: 48.5 mJ / 10 s = 4.85 mW

This is well below the 11 mW budget. Battery life: 6,660 J / 0.00485 W = 1.37 million seconds = 16 days.

Actual battery life will be lower due to Bluetooth (intermittent transmission of alerts), display updates, and haptic alerts. Target 7 days is achievable.

Decision 6: Regulatory and Validation

FDA 510(k) requires clinical validation. Plan:

Collect ECG data from 500 patients (diverse age, sex, comorbidities) in partnership with a hospital.

Annotate arrhythmias by board-certified cardiologists (ground truth).

Validate model sensitivity and specificity against ground truth.

Submit 510(k) with clinical data, software documentation (IEC 62304), and cybersecurity documentation.

Expected timeline: 18 months from prototype to market clearance.

## Final System Specification

Hardware: STM32U5, AFE (analog front-end for ECG acquisition), 500 mAh Li-ion battery

 Model: 1D CNN (8 layers, 800K parameters, int8 quantized, 800 KB)

 Inference schedule: Batch processing every 10 seconds

 Performance: 97.8% sensitivity for VT/VF, 95.2% specificity, 16-day battery life (target: 7 days, margin for Bluetooth and UI)

 Cost: $180 per unit (hardware BOM)

Alternatives rejected and why:

Smartphone processing: HIPAA risk, connectivity dependency unacceptable for medical device.

Rule-based detector: 92% sensitivity insufficient for FDA approval.

Cloud processing: Regulatory complexity (cloud infrastructure is part of medical device system, must be validated).

Case C: Agricultural Edge Sensor — Crop Disease Detection

Application Requirements

A precision agriculture company is developing a solar-powered sensor for early detection of crop diseases in vineyards. The sensor captures images of grape leaves and identifies fungal infections (powdery mildew, downy mildew) 7–10 days before visible symptoms appear, allowing targeted fungicide application instead of blanket spraying.

Functional Requirements:

Capture 224×224 RGB images of leaves every 6 hours (daylight only)

Detect 3 disease classes: healthy, powdery mildew, downy mildew

Accuracy ≥85% (false negatives cost crop yield, false positives cost unnecessary fungicide)

Transmit daily disease reports via LoRaWAN to a farm management system

Physical Constraints:

Solar-powered (2W solar panel, 5000 mAh Li-ion buffer battery)

Deployed in vineyards (no WiFi, no cellular—LoRaWAN only)

Outdoor (IP67 rated, -10°C to +50°C operating temperature)

Cost: <$150 per unit (high-volume agriculture IoT pricing)

Energy Constraints:

Solar panel output: 2W peak, but average over 24 hours (accounting for night, clouds) is ~0.5W = 500 mW

Energy storage: 5000 mAh × 3.7V = 18,500 mWh = 66,600 J

Constraint Quantification

Power budget:

 Must operate indefinitely on solar harvesting. Average power consumption must not exceed average solar input (500 mW).

## Latency budget:

 Not time-critical. Inference can take seconds or minutes—disease progression is slow.

Connectivity:

 LoRaWAN has low bandwidth (1–50 kbps) and high latency (1–10 seconds round-trip). Cannot transmit raw images (224×224×3 = 150 KB per image is too large). Must transmit only classification results.

Decision 1: Processing Architecture

Option 1: Fully on-device.

 Sensor runs inference locally. Transmits classification result (10 bytes: timestamp + class label + confidence) via LoRaWAN.

Pros: Minimal bandwidth (10 bytes vs. 150 KB). Privacy-preserving (proprietary crop data stays on farm).

Cons: Must run vision model on edge processor. Power budget must include inference.

Option 2: Cloud processing.

 Sensor compresses image (JPEG, ~20 KB) and transmits via LoRaWAN. Cloud server runs inference.

Pros: Can use large, accurate models (ResNet50, EfficientNet-B3).

Cons: LoRaWAN bandwidth is insufficient for 20 KB images every 6 hours (takes 3–5 minutes to transmit one image at 1 kbps). Latency is unacceptable.

Option 3: Edge gateway processing.

 Sensor transmits images to a local gateway (Raspberry Pi in the vineyard office) via WiFi or LoRa. Gateway runs inference.

Pros: Gateway is mains-powered, can run larger models.

Cons: Requires gateway installation (cost). WiFi range is limited (vineyards are large). Gateway is single point of failure for all sensors.

Decision: Select Option 1 (fully on-device). Rationale:

LoRaWAN bandwidth cannot support image transmission at 6-hour intervals.

Gateway adds $200–300 cost per vineyard (amortized over 20–50 sensors, but still a deployment complexity).

Solar power budget (500 mW average) can sustain low-duty-cycle inference (image capture + inference 4× per day).

## Decision 2: Model Selection

Crop disease detection from leaf images is an image classification problem. Candidates:

Approach A: MobileNetV2 (pretrained on ImageNet, fine-tuned).

 Use MobileNetV2-1.0 (full size), pretrain on ImageNet, fine-tune final layers on grape disease dataset.

Pros: High accuracy (90%+ in similar agricultural tasks). Well-supported by TFLite.

Cons: Large (14 MB float32, 3.5 MB int8). Slow on embedded processors (500–1000 ms inference on Raspberry Pi, slower on MCUs).

Approach B: MobileNetV2-0.35 (scaled down).

 Use a much smaller MobileNetV2 variant (35% width multiplier), train from scratch on grape disease dataset.

Pros: Small (1.6 MB float32, 400 KB int8). Fast (80–150 ms on Cortex-A class processors).

Cons: Lower accuracy (82–88% typical). May not distinguish subtle early-stage disease symptoms.

Approach C: EfficientNet-Lite0.

 Smallest EfficientNet variant, optimized for mobile/edge.

Pros: Better accuracy than MobileNetV2-0.35 for similar model size (88–92% on agricultural datasets).

Cons: Slightly larger (4.6 MB int8). Requires more compute (350–500 ms inference).

Approach D: Custom lightweight CNN.

 Design a task-specific architecture (5–7 convolutional layers, no residual connections).

Pros: Smallest possible (200–600 KB int8). Fastest (50–100 ms).

Cons: Lower accuracy (78–84%). Requires architecture design expertise.

## Decision: Select Approach C (EfficientNet-Lite0). Rationale:

Accuracy is critical (false negatives = lost crop, false positives = wasted fungicide).

88–92% accuracy is achievable with EfficientNet-Lite0, which meets the 85% threshold with margin.

Model size (4.6 MB int8) and inference time (350–500 ms) are acceptable given low duty cycle (4 inferences per day).

Decision 3: Hardware Selection

Constraints:

Flash: 5 MB model + firmware

SRAM: 1–2 MB (activations for 224×224 images)

Power: Average 500 mW (solar input), peak during inference can be higher (battery buffer)

Must include camera, LoRaWAN radio, solar charge controller

Cost: <$150 total

Candidates:

Option 1: Raspberry Pi Zero 2 W

Quad-core Cortex-A53 at 1 GHz

512 MB RAM

Supports camera via CSI, LoRaWAN via SPI module

Power: 150–400 mW idle, 1,000–2,000 mW during inference

Cost: $15 (Pi) + $25 (camera) + $30 (LoRaWAN module) + $40 (solar + battery + enclosure) = $110

Option 2: STM32H7 + external camera

Cortex-M7 at 480 MHz

1 MB SRAM, 2 MB flash (insufficient for 4.6 MB model—need external flash)

Power: 50 mW sleep, 500 mW during inference

Cost: $8 (MCU) + $40 (camera module with image sensor) + $30 (LoRaWAN) + $40 (solar + battery) = $118

Option 3: ESP32-S3 with camera

Dual-core Xtensa at 240 MHz

512 KB SRAM, 8 MB flash

Built-in camera interface

Power: 10 µW sleep, 300 mW during inference

Cost: $5 (ESP32-S3) + $15 (camera module) + $30 (LoRaWAN) + $40 (solar + battery) = $90

Decision: Select Raspberry Pi Zero 2 W. Rationale:

SRAM: 512 MB is more than sufficient for EfficientNet-Lite0 activations (~5 MB).

Software ecosystem: TensorFlow Lite for Python is mature. Camera and LoRaWAN libraries are well-supported.

Power: 1–2W during inference is high, but with 4 inferences per day (each taking ~2 seconds), total inference energy is 1.5W × 2s × 4 = 12 Wh per day. Sleep energy (150 mW × 86,400 s = 3.6 Wh per day). Total: 15.6 Wh per day. Solar panel provides 0.5W × 24h = 12 Wh per day in worst case (overcast). Battery buffer (18.5 Wh) covers the deficit for 1–2 days of clouds.

ESP32-S3 is cheaper and lower power, but 512 KB SRAM is marginal for 224×224 image processing. STM32H7 lacks sufficient on-chip flash and SRAM.

## Decision 4: Inference Scheduling

Strategy: Event-driven by daylight. Use a light sensor to detect sunrise. Capture and classify images at 6 AM, 10 AM, 2 PM, 6 PM (4 per day, evenly spaced during daylight hours).

Power profile per inference event:

Wake from sleep: 1 second at 500 mW = 0.5 J

Capture image: 0.5 seconds at 800 mW = 0.4 J

Run inference: 2 seconds at 1,500 mW = 3 J

Transmit result via LoRaWAN: 3 seconds at 100 mW = 0.3 J

Return to sleep: 0.1 J

Total: 4.3 J per event

Daily energy budget:

Inference events: 4 × 4.3 J = 17.2 J

Sleep (23.9 hours): 0.15W × 86,040 s = 12,906 J

Total: 12,923 J per day = 3.6 Wh

Solar input (worst case): 12 Wh per day (overcast).

Energy balance is positive. System can operate indefinitely on solar.

## Decision 5: Model Training and Deployment

Dataset: Collect 5,000 images of grape leaves in vineyards (1,500 healthy, 1,800 powdery mildew, 1,700 downy mildew). Images captured at different times of day, different growth stages.

Training: Fine-tune EfficientNet-Lite0 (pretrained on ImageNet) on grape disease dataset. Apply data augmentation (rotation, brightness, crop).

Validation accuracy (float32): 91.2%

 Validation accuracy (int8 PTQ): 89.8%

Accuracy exceeds 85% threshold. Deploy int8 model (4.6 MB) to Raspberry Pi Zero 2 W.

Deployment:

TensorFlow Lite for Python

Camera: picamera library

LoRaWAN: pyLoRa library

Scheduler: cron job triggers inference at fixed times

Field Trial Results

Deploy 10 sensors across 2 vineyards for 3 months (growing season).

Performance:

Accuracy: 87.3% (lower than validation due to domain shift—field images have more variation than training set)

False negatives: 8% (missed infections, leading to delayed treatment)

False positives: 5% (unnecessary fungicide application)

Battery life: Indefinite (solar sustained operation through 5 consecutive overcast days)

Issues identified:

Morning dew on leaves caused image blur, reducing accuracy. Fixed by delaying first capture to 7 AM (after dew evaporates).

LoRaWAN gateway coverage gaps in parts of vineyard. Added one repeater.

## Final System Specification

Hardware: Raspberry Pi Zero 2 W, Pi Camera v2, RFM95W LoRaWAN module, 2W solar panel, 5000 mAh Li-ion battery

 Model: EfficientNet-Lite0 (int8 quantized, 4.6 MB)

 Inference schedule: 4× per day (daylight hours)

 Performance: 87% accuracy, indefinite battery life (solar-sustained)

 Cost: $110 per unit

Alternatives rejected and why:

Cloud processing: LoRaWAN bandwidth insufficient for image transmission.

MobileNetV2-0.35: Accuracy (82%) below 85% threshold.

ESP32-S3: SRAM (512 KB) too small for 224×224 image processing with EfficientNet activations.

Synthesis: What These Case Studies Teach

Each case study applied the full decision framework:

Quantify constraints (power, latency, memory, connectivity, cost).

Select processing architecture (on-device, edge, cloud, hybrid).

Select model approach (classical, decision tree, CNN, LSTM).

Select hardware (MCU, application processor, accelerator).

Optimize model (quantization, pruning, distillation).

Design inference schedule (duty cycle, event-driven, batch).

Validate on target hardware (accuracy, latency, power).

Document alternatives rejected and why.

The decisions differ across cases because the constraints differ:

Case A (industrial): Power is the binding constraint. Solution: ultra-low-power MCU, infrequent inference, decision tree model.

Case B (medical): Accuracy and regulatory requirements are binding. Solution: larger model, aggressive optimization (QAT, distillation), security-focused hardware.

Case C (agriculture): Bandwidth and solar harvesting are binding. Solution: on-device vision model on application processor, event-driven by daylight.

No single solution fits all applications. The framework is the same. The constraints determine the outcome.

You are now equipped to apply this framework to your own applications. The terminal capability—examining a real embedded design problem, assessing whether AI integration is appropriate, selecting an inference and deployment strategy, and justifying every decision against hardware constraints—is yours.

Use it.


---

*[<- Chapter 13](./chapter-13-tinyml-toolchains.md) | [Table of Contents](../README.md) | [Appendix A ->](./appendix-a-embedded-quick-reference.md)*
