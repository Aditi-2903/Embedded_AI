# Chapter 9: Communication and the Edge-Cloud Spectrum

Your model fits in memory, runs in 80 milliseconds, and consumes 12 mJ per inference. All local constraints are satisfied. But the model achieves only 87% accuracy, and your application requires 92%. A larger, more accurate model exists—2 million parameters, 150 million MACs—but it doesn't fit on your microcontroller. The device has LoRaWAN connectivity with 1 kbps uplink bandwidth and 500 ms round-trip latency to the nearest gateway.
You face a decision: accept the lower accuracy of the on-device model, or send sensor data to a cloud server that runs the larger model. If you offload to the cloud, you gain 5% accuracy but pay latency, bandwidth, energy, and privacy costs. The question is not whether the cloud is more powerful—it obviously is. The question is whether the communication cost is acceptable for your application.
This is the edge-cloud tradeoff. Embedded systems exist on a spectrum from fully local processing (all inference on-device) to fully cloud processing (device sends raw data, server does everything). Between these extremes lie hybrid strategies where computation is split between the device, an edge gateway, and the cloud. This chapter teaches you to quantify the costs of each approach and select the right processing location for your application's constraints.
By the end of this chapter, you will be able to calculate the latency, bandwidth, power, and privacy costs of offloading inference, determine the appropriate processing tier (device, edge, cloud) for a given application, and design hybrid inference strategies that distribute computation to minimize total system cost.
The Edge-Cloud Spectrum: On-Device, Near-Edge, Far-Edge, Cloud
Processing can happen at four tiers, each with different performance, cost, and constraint profiles.
On-device (endpoint): Inference runs entirely on the embedded device—microcontroller, edge processor, or sensor node. No data leaves the device except the final result (classification label, anomaly flag, detected object). This minimizes latency, preserves privacy, and requires no network connectivity. But it's constrained by the device's memory, compute, and power budgets.
Near-edge (local gateway): Inference runs on a local gateway within the same building or facility—a Raspberry Pi, industrial PC, or edge server. The device sends raw sensor data to the gateway over WiFi, Ethernet, or serial link. The gateway has more compute power than the endpoint (multi-core CPU, GPU) and more relaxed power constraints (wall-powered). Latency is low (1–50 ms round-trip). But the gateway adds cost and requires local infrastructure.
Far-edge (regional server): Inference runs on a server at a regional datacenter or cellular base station. The device sends data over cellular, LoRaWAN, or satellite link. Latency is higher (50–500 ms round-trip) and variable. Bandwidth may be limited (1 kbps to 1 Mbps). But compute is essentially unlimited—the server can run models with billions of parameters.
Cloud (centralized datacenter): Inference runs on a cloud platform (AWS, Azure, GCP). The device sends data over the internet. Latency is highly variable (100 ms to several seconds depending on network conditions). Bandwidth costs are recurring. But the cloud offers infinite scalability, continuous model updates, and access to the latest AI models.
The tradeoff surface has four dimensions:
Latency: On-device is fastest (0 ms communication latency, only inference latency). Cloud is slowest (round-trip network latency dominates). Hard real-time applications (collision avoidance, industrial control) cannot tolerate cloud latency. Soft real-time applications (smart home, wearables) might accept it.
Privacy: On-device is most private (data never leaves the device). Cloud is least private (raw data crosses the internet and is processed by a third party). Medical devices, security cameras, and financial sensors often have regulatory requirements that preclude cloud processing.
Connectivity: On-device requires no connectivity. Cloud requires always-on internet. Far-edge requires intermittent connectivity. Applications in remote locations (agriculture, wildlife monitoring, maritime) may have no reliable network.
Cost: On-device has no recurring communication cost but limited compute. Cloud has recurring bandwidth and API costs but unlimited compute. Near-edge has upfront gateway cost but no recurring fees. The economic breakeven depends on deployment scale and lifetime.
The decision workflow is:
Identify the application's latency requirement. If latency < 100 ms, consider on-device or near-edge.
Identify privacy and regulatory constraints. If data cannot leave the device or facility, reject cloud and far-edge.
Identify connectivity availability. If the device has no network or intermittent connectivity, reject cloud.
Compare compute requirements to device capabilities. If the model fits on-device, prefer on-device. If not, evaluate offloading.
Calculate communication costs (latency, bandwidth, energy, recurring fees). If acceptable, offload. If not, reduce the model or upgrade the device.
There is no universal answer. A smart thermostat with WiFi, wall power, and relaxed latency can use cloud inference cheaply. A battery-powered vibration sensor in a remote oilfield with cellular connectivity and hard latency requirements cannot.
Communication Cost as a Constraint
Communication costs four resources: latency, bandwidth, energy, and money. Each must be quantified to determine whether offloading is viable.
Latency Cost
Round-trip latency is the time between sending data and receiving a result. It includes:
Packetization and serialization (converting sensor data to a transmittable format)
Transmission time (data size / bandwidth)
Network propagation delay (speed-of-light limit, router hops)
Server queueing and processing time (inference on the remote server)
Return transmission time
For a LoRaWAN device sending 100 bytes of sensor data to a server and receiving a 10-byte result:
Serialization: 5 ms
LoRa uplink (SF7, 125 kHz, effective 5 kbps): 100 bytes / (5000 bits/s / 8) = 160 ms
Network propagation (gateway to server): 50 ms
Server inference: 20 ms (fast server, small model)
LoRa downlink: 10 bytes / (5000 bits/s / 8) = 16 ms
Deserialization: 2 ms
Total round-trip latency: 253 ms
If the application requires results within 100 ms, cloud inference fails. The network latency alone (160 + 50 + 16 = 226 ms) exceeds the budget before inference even starts.
Bandwidth Cost
Bandwidth determines how much data can be sent per unit time. Embedded communication links are often bandwidth-constrained:
LoRaWAN: 0.3–50 kbps (depending on spreading factor and region)
NB-IoT: 20–250 kbps
LTE Cat-M1: 1 Mbps (uplink and downlink)
WiFi: 1–100 Mbps (highly variable)
Sending raw sensor data consumes bandwidth proportional to the data size. A 96×96 RGB image is 27,648 bytes. On LoRaWAN at 5 kbps, transmission takes:
T = 27,648 bytes × 8 bits/byte / 5000 bits/s = 44 seconds
That's unusable for real-time applications. Even on WiFi at 1 Mbps, transmission takes 0.22 seconds—acceptable for some applications but not for others.
The bandwidth constraint is satisfied if:
Data size / Bandwidth < Latency budget
If your latency budget is 500 ms and your bandwidth is 10 kbps, the maximum data size is:
Max data = 0.5 s × (10,000 bits/s / 8 bits/byte) = 625 bytes
A 96×96 image doesn't fit. A 32×32 image (3,072 bytes) doesn't fit. But a 16-element feature vector (64 bytes) does fit.
This is why edge processing often includes feature extraction on-device. Instead of sending raw images, the device computes a low-dimensional feature vector (histogram, edge map, statistical features) and sends that. The server receives 64 bytes instead of 27,648 bytes, and bandwidth usage drops by 430×.
Energy Cost
Wireless transmission consumes energy proportional to data size and transmission power. For a LoRaWAN device transmitting at +14 dBm:
Transmit current: 40 mA
Transmission time: 160 ms (for 100 bytes at SF7)
Energy per transmission: 3.3 V × 0.04 A × 0.16 s = 21.1 mJ
If the device transmits every 10 minutes (144 times per day), daily communication energy is:
E_comm = 21.1 mJ × 144 = 3,038 mJ = 3.04 J per day
For a 2000 mAh battery at 3.3 V (6600 mWh = 23,760 J), communication energy is:
3.04 J / 23,760 J = 0.013% of battery capacity per day
Communication energy is negligible compared to sleep energy (3.3 V × 0.01 mA × 24 hours = 0.79 J per day) and inference energy (if running locally).
But if the device transmits every 10 seconds instead of every 10 minutes, transmissions per day increase to 8,640. Daily communication energy becomes 182 J—nearly 8× the total battery capacity. The battery drains in 3 hours.
The energy constraint for communication is:
E_comm_per_event × Events_per_day < Power_budget × 86,400 s
If your power budget is 1 mW (86.4 J per day) and communication costs 21 mJ per event, the maximum sustainable event rate is:
Max events = 86.4 J / 0.021 J = 4,114 events per day = 1 event every 21 seconds
More frequent communication requires a larger battery, energy harvesting, or reduced transmission power (which reduces range).
Monetary Cost
Cloud inference has recurring costs: bandwidth (data transmission) and compute (inference API calls). Cellular connectivity has monthly data plans. Cloud API providers (AWS SageMaker, Google Vertex AI, Azure ML) charge per inference or per compute-hour.
For a cellular IoT device on an NB-IoT plan:
Data plan: $5/month for 10 MB
Sensor data per transmission: 100 bytes
Transmissions per month (every 10 minutes): 4,320
Total data: 432 KB per month (well within 10 MB)
For cloud inference via AWS SageMaker:
Inference cost: $0.0001 per inference (example pricing)
Inferences per month: 4,320
Monthly cost: $0.43
Total monthly cost per device: $5.43.
For 10,000 deployed devices, annual cost is:
10,000 devices × $5.43/month × 12 months = $651,600 per year
That's $651,600 in recurring operational expense. Compare to on-device inference:
Upfront hardware cost: $5 more per device for a faster processor
Total upfront cost: $50,000
Recurring cost: $0
The crossover point is less than one year. On-device inference has higher upfront cost but lower lifetime cost. For long-lived deployments (5–10 years), on-device is cheaper. For short-lived deployments or prototypes, cloud is cheaper.
Privacy and Regulatory Constraints on Data Transmission
Some data cannot be transmitted, regardless of cost. Medical records are governed by HIPAA in the US and GDPR in Europe. Financial transactions are governed by PCI-DSS. Industrial control data may be subject to NIST or IEC standards. Sending raw data to the cloud may violate these regulations even if the user consents.
For medical wearables, the decision tree is:
Can the data be anonymized and aggregated? If yes, cloud processing might be allowed.
Is the data diagnostic (used for clinical decisions)? If yes, FDA or CE Mark approval may be required, which complicates cloud deployments.
Can the data be processed locally and only summary statistics sent to the cloud? If yes, this is the preferred architecture.
For industrial sensors, the decision tree is:
Is the data proprietary (trade secrets, process parameters)? If yes, on-premise processing is required.
Is the facility connected to the public internet? If no, cloud processing is physically impossible.
Does the application have safety implications (IEC 61508)? If yes, cloud latency may violate safety requirements, and local processing is mandatory.
Privacy is not just a regulatory constraint—it's also a user trust constraint. A home security camera that sends video to the cloud faces user resistance even if technically legal. On-device processing addresses the trust gap: "your data never leaves your home."
The privacy-preserving architecture is:
Run inference locally.
Send only the result (classification label, anomaly flag) to the cloud.
Store raw data locally or discard it.
This satisfies most privacy regulations and user expectations. But it requires the local device to be powerful enough to run the model, which brings us back to the hardware constraint from Chapter 8.
Latency Requirements That Preclude Cloud Inference
Some applications have latency budgets that cannot be met with network communication, even on fast networks. These are hard constraints—no amount of optimization will make cloud inference viable.
Collision avoidance (autonomous vehicles, drones): Detection to action in under 50 ms. Even on a local 5G network with 10 ms round-trip latency, adding server inference time pushes total latency over 50 ms. Local processing is mandatory.
Industrial robotics (pick-and-place, quality inspection): 10–100 ms latency budgets depending on line speed. A WiFi round-trip (10–20 ms) plus server inference (10–50 ms) might fit, but network jitter can cause violations. Local processing avoids non-determinism.
Medical alarms (arrhythmia detection, seizure detection): 1–5 second budgets allow cloud processing in theory, but reliability constraints (what if the network fails?) make local processing the safer choice.
Human-computer interaction (voice assistants, AR/VR): 50–200 ms latency budgets for responsiveness. Cloud voice assistants (Alexa, Google Assistant) accept this latency. Local voice assistants (Apple Siri on-device mode) are faster but less accurate.
The latency budget determines the maximum allowable network round-trip time. If your application requires 100 ms end-to-end and inference takes 40 ms, network latency cannot exceed 60 ms. On WiFi or Ethernet, this is feasible. On cellular or LoRaWAN, it's not.
Split Inference: Running Early Layers On-Device, Later Layers on a Server
Split inference is a hybrid strategy where the model is partitioned: early layers run on the device, later layers run on a server. The device sends intermediate activations (not raw data) to the server, which completes inference and returns the result.
The advantage is reduced bandwidth and improved privacy. Intermediate activations are smaller and less interpretable than raw inputs. A 96×96 image is 27 KB, but the activations after three convolutional layers might be 24×24×64 = 36 KB as float32 or 9 KB as int8. Even better, the activations don't resemble the original image—they're feature maps, not pixels—so privacy is partially preserved.
The split point is chosen based on:
Bandwidth constraint: Split where activation size drops below bandwidth capacity.
Compute constraint: Offload the most compute-intensive layers to the server.
Privacy constraint: Run enough layers on-device to obscure the original input.
A worked example: image classification on a resource-constrained edge camera.
Model architecture (MobileNetV2):
Input: 224×224×3 (150 KB)
After Conv1: 112×112×32 (400 KB as float32, 100 KB as int8)
After block 3: 28×28×96 (75 KB as float32, 19 KB as int8)
After block 6: 14×14×160 (31 KB as float32, 8 KB as int8)
After block 13: 7×7×320 (16 KB as float32, 4 KB as int8)
Final output: 1000 classes (4 KB as float32)
Deployment strategy:
On-device (ESP32-S3 with WiFi): Run up to block 6 (first 60% of operations, 30 MB MACs). Send 8 KB of int8 activations to the server.
On-server (cloud API): Run blocks 7–13 and classification head (remaining 40%, 20 MB MACs). Return 1000-class probability vector (4 KB).
Analysis:
Bandwidth: 8 KB uplink, 4 KB downlink = 12 KB total. On WiFi (1 Mbps), transmission time is 96 ms.
Latency: On-device (180 ms) + WiFi transmission (96 ms) + server inference (20 ms) + WiFi return (32 ms) = 328 ms.
Compare to full on-device: 450 ms (ESP32 cannot run the full model efficiently). Split inference is faster.
Compare to full cloud: Raw image transmission (150 KB / 1 Mbps = 1200 ms). Split inference is 3.7× faster.
Split inference works when:
The device can run part of the model but not all of it.
Intermediate activations are smaller than the raw input.
Latency budget allows for round-trip network communication.
It fails when:
Bandwidth is too low even for activations (e.g., LoRaWAN cannot handle 8 KB per inference).
Network latency dominates total latency (split inference is no faster than full cloud).
Privacy requires that no data, not even activations, leaves the device.
Federated Learning Basics: Training at the Edge Without Transmitting Raw Data
Federated learning is a training paradigm where models are trained across many devices without centralizing data. Each device trains locally on its data, computes model updates (gradients or weight deltas), and sends only the updates to a central server. The server aggregates updates from all devices and distributes the improved model back to the devices.
For embedded AI, federated learning solves two problems:
Privacy: Raw data never leaves the device. The server sees only aggregated gradients, which are less sensitive than raw inputs.
Bandwidth: Sending weight updates (a few MB) is cheaper than sending all training data (gigabytes).
But federated learning is a training-time technique, not an inference-time technique. It's relevant when:
You have deployed devices generating data continuously.
You want to improve the model over time using that data.
Privacy or bandwidth constraints prevent sending raw data to a server.
Federated learning is not covered in detail in this book (which focuses on inference), but it's conceptually important because it changes the edge-cloud division of labor. Instead of "device does inference, cloud does training," federated learning enables "device does both inference and local training, cloud only aggregates."
For embedded AI designers, the takeaway is: if your deployment includes thousands of devices and you want the model to improve from field data, federate learning is an option. But it requires devices with enough compute and power to run training (not just inference), which is a much higher bar.
A Worked Example: Three Deployment Strategies for Predictive Maintenance
You are deploying a predictive maintenance system for industrial motors. The system monitors vibration data from accelerometers and predicts bearing faults 24 hours before failure. You have three models:
Model A (small): 50,000 parameters, 8 million MACs, 82% fault detection accuracy.
 Model B (medium): 200,000 parameters, 40 million MACs, 91% accuracy.
 Model C (large): 1 million parameters, 200 million MACs, 96% accuracy.
Your requirements:
Accuracy: ≥90% (Model B or C required)
Latency: ≤1 second (soft constraint—maintenance alerts are not time-critical)
Connectivity: Cellular (NB-IoT, 20 kbps, 200 ms round-trip latency)
Privacy: Data is proprietary but not regulated—cloud processing is allowed
Cost: <$10/device/month for communication and compute
You evaluate three deployment strategies.
Strategy 1: Full On-Device (Model B)
Hardware: STM32H7 (Cortex-M7 at 480 MHz)
 Inference latency: 220 ms (measured on target)
 Communication: Send only fault prediction (1 byte) once per day
 Monthly data usage: 30 bytes (negligible)
 Monthly cost: $2 (minimum NB-IoT plan)
 Accuracy: 91%
Analysis: Meets all constraints. Accuracy is at the minimum threshold. Latency is well within budget. Cost is low. This is the baseline solution.
Strategy 2: Full Cloud (Model C)
Hardware: STM32L4 (low-power MCU, sensor acquisition only)
 Data sent: 1 kHz accelerometer data, 60-second windows = 180 KB per transmission
 Transmissions per day: 24 (hourly monitoring)
 Monthly data usage: 180 KB × 24 × 30 = 129.6 MB
 NB-IoT cost: $15/month for 100 MB plan (129.6 MB exceeds 100 MB—need 200 MB plan at $25/month)
 Cloud API cost: 24 inferences/day × 30 days × $0.001/inference = $0.72/month
 Total monthly cost: $25.72
 Latency: 200 ms (network) + 50 ms (server inference) = 250 ms
 Accuracy: 96%
Analysis: Exceeds cost budget by 2.6×. Accuracy is higher (+5% over on-device), but the cost increase is not justified for this application. Bandwidth usage is also problematic—200 MB per device × 1,000 devices = 200 GB/month of cellular data.
Strategy 3: Split Inference (Model C, first 5 layers on-device)
Hardware: STM32H7 (same as Strategy 1)
 On-device: First 5 layers (30 million MACs, 120 ms latency)
 Activation size after layer 5: 4 KB (int8)
 Data sent: 4 KB per inference
 Transmissions per day: 24
 Monthly data usage: 4 KB × 24 × 30 = 2.88 MB
 NB-IoT cost: $5/month for 10 MB plan
 Cloud API cost: $0.72/month (same as Strategy 2)
 Total monthly cost: $5.72
 Latency: 120 ms (device) + 200 ms (network) + 30 ms (server) = 350 ms
 Accuracy: 96%
Analysis: Meets cost budget. Accuracy matches Model C. Latency is acceptable (350 ms < 1 second budget). Bandwidth usage is 45× lower than full cloud (2.88 MB vs. 129.6 MB). This is the optimal solution.
Conclusion:
Split inference (Strategy 3) achieves the best balance: Model C's 96% accuracy, $5.72/month cost (within budget), and 350 ms latency (within budget). Full cloud (Strategy 2) has the same accuracy but costs 4.5× more. Full on-device (Strategy 1) is cheaper but sacrifices 5% accuracy.
The edge-cloud decision is not binary. Hybrid strategies often outperform either extreme.
When to Reject Communication Entirely
Some applications cannot rely on network connectivity:
Intermittent connectivity: Underwater sensors, remote agriculture, disaster response robots. The network is unavailable most of the time.
Mission-critical reliability: Medical implants, industrial safety systems. Network failures cannot be tolerated—local processing is the only option.
Adversarial environments: Military and security applications where the network may be jammed or compromised.
For these applications, on-device inference is not an optimization—it's a requirement. The model must fit on the device, even if that means accepting lower accuracy or upgrading to more expensive hardware.
What Comes Next
The edge-cloud spectrum addresses physical and network constraints. But it assumes inference succeeds in a timely, predictable manner. The next chapter examines the hardest constraint category: real-time guarantees. When inference must meet hard deadlines with provable certainty, neural networks become difficult to integrate into safety-critical systems. Chapter 10 teaches you to evaluate real-time AI and apply design patterns that reduce risk.

---

*[← Chapter 8](./chapter-08-hardware-for-ai.md) | [Table of Contents](../README.md) | [Chapter 10 →](./chapter-10-real-time-ai.md)*