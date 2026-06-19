# **Automated Photo Identification and Classification for Traffic Violations Using Computer Vision**

## **Executive Summary**

The escalating density of urban vehicular traffic, particularly in heterogeneous and unstructured environments, necessitates the deployment of automated, highly scalable enforcement mechanisms. Traditional manual monitoring of traffic intersections is operationally inefficient, prone to human error, and economically unviable at a municipal scale. The objective of this report is to architect a practical, cost-effective, and deployable computer vision prototype for automated traffic violation detection. Moving beyond purely academic frameworks that prioritize marginal accuracy improvements over computational efficiency, this design prioritizes real-world applicability. The proposed system leverages a modular pipeline consisting of high-speed object detection, lightweight multi-object tracking, deterministic spatial rule engines, and targeted Optical Character Recognition (OCR). This approach ensures high processing throughput on edge devices, minimizes cloud bandwidth dependency, circumvents restrictive commercial software licenses, and generates legally defensible, annotated evidence.

## **Image Preprocessing and Environmental Normalization**

Before neural networks can accurately detect vehicles or optical character recognition engines can decipher license plates, the raw optical input must be normalized to account for extreme environmental variations. Traffic surveillance cameras operate continuously, subjecting the system to fluctuating illumination, precipitation, and motion artifacts.  
To handle low-light conditions and shadows, the pipeline must implement dynamic histogram equalization, specifically Contrast Limited Adaptive Histogram Equalization (CLAHE). Unlike standard histogram equalization, which amplifies noise in uniformly dark regions, CLAHE operates on localized image tiles, enhancing the contrast of vehicle features and license plates without oversaturating illuminated areas like headlights.  
During periods of rain or fog, the optical feed suffers from reduced high-frequency spatial details. The implementation of lightweight spatial filters, such as bilateral filtering, preserves the critical edge structures of vehicles while smoothing out the high-frequency noise introduced by precipitation. Motion blur, a frequent challenge when capturing high-speed vehicles, requires the physical camera shutter speed to be configured optimally (typically 1/1000 of a second or faster). However, software-side algorithmic deblurring using Wiener filters can be applied selectively to the cropped regions of interest containing the license plates, ensuring the OCR module receives the sharpest possible input without burdening the entire frame's processing pipeline.

## **Implementation Architecture Evaluation**

Designing a system capable of processing high-resolution video streams in real-time requires a careful balance between deep learning perception and algorithmic logic. Three architectural paradigms are evaluated for their suitability in a municipal deployment.

### **Option A: Object Detection \+ Tracking \+ Rule Engine \+ OCR**

This modular architecture utilizes a primary deep learning model to locate vehicles, riders, and accessories within a 2D frame. A kinematic tracking algorithm associates these detections temporally across sequential frames. A deterministic, non-neural rule engine utilizes the resulting spatial trajectories to infer violations based on predefined virtual loops or geofenced polygons. Upon detecting a violation, the system isolates the license plate and invokes the OCR module solely on that localized crop.  
This architecture represents the industry standard for commercial traffic enforcement systems. Its primary advantage is computational efficiency; the heavy deep learning inference is restricted to a single unified detection pass, while the violation logic is handled by computationally trivial geometric mathematics. Furthermore, false positives can be debugged geometrically without necessitating the retraining of the neural network. By invoking the OCR module exclusively during a triggered violation event, the system conserves massive amounts of processing power.

### **Option B: Multiple Specialized Deep-Learning Models**

An alternative approach involves cascading several specialized neural networks. A primary model detects vehicles, a secondary model cropped on the windshield classifies seatbelt usage, a tertiary model identifies traffic light states, and a complex spatiotemporal 3D Convolutional Neural Network (3D-CNN) processes sequences of frames to classify behavioral violations like wrong-way driving.  
While this approach can achieve high theoretical accuracy on isolated tasks, it is computationally prohibitive for a deployable edge prototype. Running multiple deep networks concurrently overwhelms the memory bandwidth and compute capacity of modest hardware, dropping the processing rate well below real-time thresholds. Furthermore, relying on 3D-CNNs for behavioral violations introduces a "black box" logic that lacks the legal explainability required for traffic citations.

### **Option C: Vision-Language Models**

The advent of multimodal foundation models allows systems to process a video frame and respond to natural language prompts, such as querying whether a rider is wearing a helmet.  
While Vision-Language Models (VLMs) offer impressive zero-shot capabilities and eliminate the need for extensive custom dataset curation, they are catastrophically slow for real-time video processing. Even highly optimized VLMs operating on top-tier GPUs struggle to exceed single-digit frames per second, making it impossible to track fast-moving vehicles or capture fleeting violations. The latency and high parameter count render VLMs unsuitable for cost-effective edge deployment.

### **Recommended Architecture**

The deployment dictates the selection of **Option A**. Isolating deep learning to purely visual perception tasks while delegating violation inference to deterministic algorithms provides the most robust, scalable, and explainable path to a Minimum Viable Product (MVP).

## **Technology Stack Selection and Justification**

The selection of the underlying technology stack directly impacts the long-term scalability, deployment cost, and commercial viability of the prototype.

| Component | Evaluated Options | Recommended Choice | Technical Justification |
| :---- | :---- | :---- | :---- |
| **Object Detection** | YOLOv11, YOLOv8, RT-DETR, YOLOv10 | **RT-DETR** or **YOLOv10** | YOLOv8 and YOLOv11 operate under the AGPL-3.0 license, requiring commercial users to open-source their entire software stack or purchase expensive enterprise licenses. RT-DETR (Baidu) and YOLOv10 (Tsinghua University) are available under the permissive Apache 2.0 license, enabling restriction-free commercial deployment. Furthermore, RT-DETR and YOLOv10 eliminate the need for Non-Maximum Suppression (NMS) post-processing, significantly stabilizing inference latency on edge devices. |
| **Multi-Object Tracking** | DeepSORT, ByteTrack | **ByteTrack** | DeepSORT relies on deep appearance embeddings (ReID) to associate objects, adding 10–30 milliseconds of latency per frame. ByteTrack relies purely on motion kinematics, utilizing both high and low-confidence bounding boxes to maintain tracks through occlusions, adding less than 5 milliseconds of overhead. For traffic scenarios where vehicles follow predictable paths, appearance embeddings are an unnecessary computational burden. |
| **Optical Character Recognition** | EasyOCR, PaddleOCR | **PaddleOCR** | Empirical evaluations demonstrate that EasyOCR achieves approximately 38% accuracy on challenging license plates, while PaddleOCR consistently achieves over 85%. PaddleOCR's modular architecture, encompassing text detection, directional classification, and recognition, proves far superior at handling the unstructured fonts, multiple lines, and skew inherent in global and Indian license plates. |
| **Backend Framework** | Flask, FastAPI | **FastAPI** | FastAPI's asynchronous architecture handles I/O-bound operations natively. In a video processing pipeline, asynchronous handling ensures that network requests, database writes, and RTSP stream fetching do not block the core inference loop. |
| **Frontend Dashboard** | Streamlit, React | **Streamlit** (MVP), **React** (Production) | Streamlit allows for the rapid visualization of bounding boxes, video streams, and tabular data without writing boilerplate UI code, ideal for a 4–8 week MVP. For the production roadmap, migrating to a React-based architecture provides the component lifecycle management necessary for a scalable Command and Control dashboard. |
| **Database Management** | MongoDB, PostgreSQL | **PostgreSQL** | Traffic violations are inherently relational and geospatial. PostgreSQL, augmented with the PostGIS extension, allows for robust relational mapping of vehicle logs to evidentiary images, while enabling future scalability for spatial analytics, heatmaps, and geofencing across a smart-city grid. |

## **Traffic Violation Detection Logic and Strategy**

A critical realization in intelligent transportation systems is that not all violations require specialized artificial intelligence models. Attempting to train a neural network to recognize "wrong-way driving" as a static visual class is an exercise in over-engineering. Instead, violations are divided into two categories: spatial rule-based violations and visual classification violations.

### **Rule-Based Violations**

The industry standard approach for commercial traffic enforcement systems relies on "virtual loops" and Region of Interest (ROI) geofencing. These algorithms rely exclusively on standard vehicle detection classes (Car, Truck, Bus, Motorcycle), the ByteTrack tracking algorithm, and geometric mathematics.

| Violation Type | Implementation Approach | Complexity | Real-World Accuracy |
| :---- | :---- | :---- | :---- |
| **Wrong-Side Driving** | **Tracking \+ Rule-Based Logic:** A directional vector is defined for the authorized flow of the lane. The system monitors the trajectory vector of the tracked vehicle centroid over multiple frames. A mathematical dot product compares the trajectory vector against the lane vector. If the resulting angle exceeds a predefined threshold (e.g., 90 degrees), a violation is flagged. | Easy | Very High. Accuracy is limited solely by the stability of the tracking algorithm. |
| **Stop-Line Violation** | **Tracking \+ Rule-Based Logic \+ Signal State:** A virtual polygon (ROI) is plotted immediately past the physical stop line on the video frame. The system interfaces with the traffic light controller. When the signal is red, if the bottom-center coordinate of a tracked vehicle's bounding box intersects the virtual ROI, a violation is triggered. | Easy | Exceptional. Highly deterministic and mathematically provable. |
| **Red-Light Violation** | **Tracking \+ Rule-Based Logic \+ Signal State:** This extends the stop-line logic. A secondary, deeper virtual loop is established in the center of the intersection. If a vehicle track passes through the primary stop-line ROI and subsequently enters the secondary intersection ROI while the light remains red, the system generates evidence. | Medium | High. Requires precise synchronization with the intersection's signal timing data. |
| **Illegal Parking** | **Tracking \+ Geofencing:** A "No Parking" geofenced polygon is drawn onto the frame. The system monitors the centroids of tracked vehicles. If a centroid remains within the designated ROI and its spatial displacement over a significant time threshold (e.g., 60 to 120 seconds) falls below a predefined pixel drift allowance, the vehicle is flagged as illegally parked. | Medium | High. The primary failure mode involves severe occlusion by larger passing vehicles temporarily breaking the track. |
| **Lane Violation** | **Tracking \+ Rule-Based Logic:** The solid lane markings are defined as mathematical line segments within the system coordinates. Ray-casting algorithms determine if the coordinates of the vehicle bounding box intersect the solid line segments during tracking. | Medium | Moderate. Highly susceptible to perspective distortion and requires precise camera calibration. |

### **AI-Driven Visual Violations**

These violations cannot be solved by spatial geometry alone and require specific classes to be trained into the RT-DETR or YOLO object detection model.

| Violation Type | Implementation Approach | Complexity | Real-World Accuracy |
| :---- | :---- | :---- | :---- |
| **Helmet Non-Compliance** | **Object Detection \+ Spatial Association:** The model is trained on classes: Motorcycle, Rider, Helmet, and No-Helmet. The system uses bounding box Intersection over Union (IoU) to spatially associate a detected rider with a specific motorcycle. If the No-Helmet class is detected within the bounds of an associated rider, a violation is flagged. | Medium | High (\~85-90%). Public datasets are abundant, though performance drops in dense, overlapping traffic. |
| **Triple Riding** | **Object Detection \+ Spatial Association:** Utilizing the same classes as helmet detection, the system isolates a tracked Motorcycle bounding box. It then counts the number of Rider bounding boxes that exhibit an IoU overlap greater than 30% with that motorcycle. If the count exceeds two, the system triggers a violation. | Medium | Moderate to High. Susceptible to perspective occlusion where riders are hidden behind one another relative to the camera angle. |
| **Seatbelt Non-Compliance** | **Specialized Object Detection:** The model requires targeted classes: Windshield, Seatbelt, and No-Seatbelt. This approach demands specialized camera angles facing directly into oncoming traffic. | Hard | Low to Moderate. This relies heavily on hardware interventions, such as cross-polarized lenses to defeat windshield glare and infrared (IR) flashes for cabin penetration. Relying purely on standard CCTV footage yields high false-negative rates. |
| **Mobile Phone Usage** | **Specialized Object Detection:** Similar to seatbelt detection, this requires high-resolution views into the vehicle cabin to detect a Phone-Usage class. | Hard | Low. Severe occlusion issues (hands or steering wheels blocking the device) and visual ambiguity (e.g., a driver scratching their ear) lead to unacceptable false-positive rates in automated deployment. |

**Recommended MVP Implementation:** A practical 4-8 week MVP must focus on the violations that provide the highest deterministic accuracy with the least required custom hardware. Therefore, the MVP should be strictly limited to Helmet Non-compliance, Triple Riding, Wrong-Side Driving, Stop-Line, and Red-Light violations. Seatbelt and Mobile Phone detection should be explicitly deferred to a phase-two production roadmap due to their absolute reliance on specialized optical hardware and challenging data acquisition.

## **Comprehensive Dataset Research and Survey**

The efficacy of the perception layer is inextricably linked to the quality and contextual relevance of its training data. For unstructured environments characteristic of Indian or Southeast Asian roadways, conventional Western datasets (such as COCO or Cityscapes) perform inadequately due to the absence of diverse vehicle typologies, such as auto-rickshaws or heavily laden two-wheelers, and the prevalence of chaotic lane behavior.

| Target Application | Recommended Dataset | Source / Access | Dataset Size | Key Classes / Annotations | Indian Road Suitability | Licensing / Research Usage |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **Vehicle Detection** | DriveIndia | TiHAN-IIT Hyderabad | 66,986 images | 24 classes including autorickshaws, pedestrians, heterogeneous vehicles. | Exceptional. Specifically built for unstructured Indian traffic. | Open release via TiHAN portal. Highly active in current research. |
| **Vehicle Detection** | Indian Driving Dataset (IDD) | IIIT Hyderabad / Kaggle | 41,962 labeled images | 15 classes including rider, motorcycle, autorickshaw, animal. | Exceptional. The foundational benchmark for unstructured traffic. | CC BY-NC-SA 4.0. Widely cited in autonomous navigation research. |
| **Helmet Detection** | Roboflow Helmet Kaggle 5K | Roboflow Universe | 5,000 images | Helmet, No-Helmet, Head. | High. Requires augmentation with local background environments. | CC BY 4.0. Heavily utilized in open-source prototypes. |
| **Seatbelt Detection** | Roboflow Seatbelt Datasets | Roboflow Universe | \~1,100 images | Seatbelt, No-Seatbelt. | Low to Moderate. Heavily dependent on specific camera angles and lighting. | MIT / CC BY. Often requires extensive custom data collection to be viable. |
| **Mobile Phone Usage** | DC Labs Mobile Phone Dataset | DataCluster Labs | 3,000+ images | Smartphone usage, feature phone usage. | Moderate. Crowdsourced in India, but often lacks the specific windshield-penetration perspective needed. | Commercial inquiry required. Limits open-source deployment. |
| **Mobile Phone Usage** | Roboflow Driver Distraction | Roboflow Universe | \~5,000 images | Driver talking on phone, texting, safe driving. | Moderate. Diverse angles, but suffers from false-positive ambiguity. | CC BY 4.0. Active in research for interior cabin monitoring. |
| **Number Plate Recognition** | Roboflow Indian License Plate Dataset | Roboflow Universe | Varies by fork | Alphanumeric characters, bounding boxes for plates. | High. Specifically curated for the aspect ratios of Indian plates. | CC BY 4.0. Essential for training the localization step before OCR. |
| **Traffic Surveillance** | IDD-3D | IIIT Hyderabad | 12,000 annotated frames | 3D bounding boxes for vehicles in chaotic traffic. | Exceptional. Captured via LiDAR and multi-camera setups in Hyderabad. | Academic/Research use. Helps establish spatial relationships. |
| **Wrong-Side Driving** | Wrong Way Driving Detection | Roboflow Universe | \~2,500 images | Right-side, Wrong-side. | Low. As established, wrong-way driving is a kinematic tracking problem, not a static visual classification problem. | CC BY 4.0. Not recommended for use; use rule-based logic instead. |
| **Stop-Line / Red-Light** | Sacramento Red Light Data | Data.gov / DOT | \~47,000 records | Tabular data of violations, not visual bounding boxes. | N/A. Useful for statistical modeling of violator behavior, not model training. | Public Domain. Stop-line violations should be solved via virtual loops, avoiding datasets. |
| **Illegal Parking** | Mobiusi Illegal Parking | Hugging Face | \<1,000 images | Vehicles parked on sidewalks, perpendicular parking. | Low. Small size and generalized scenes. Parking should be solved via geofencing and tracking. | CC-BY-NC-SA 4.0. Not recommended for deployment. |

**Custom Data Collection Requirements:** While DriveIndia and IDD establish a formidable baseline for vehicle detection, custom data collection is strictly required. Models trained predominantly on dashcam perspectives (like IDD) will suffer accuracy degradation when subjected to the high-angle, overhead perspectives typical of municipal CCTV cameras. Approximately 2,000 to 5,000 frames captured from the specific mounting angles of the target deployment intersections must be annotated to fine-tune the baseline models.

## **Real-Time Data Ingestion and Camera Infrastructure**

The continuous ingestion of stable video streams is a critical engineering bottleneck. The system must interface seamlessly with disparate hardware protocols.

| Ingestion Method | Availability | Cost | Ease of Deployment | Suitability for Prototype |
| :---- | :---- | :---- | :---- | :---- |
| **RTSP Streams** | Ubiquitous in modern IP cameras. | Low (Standard feature) | Easy. Natively supported by OpenCV and FFmpeg. | **Exceptional.** The Real-Time Streaming Protocol is the industry standard for transporting low-latency H.264/H.265 video over IP networks. |
| **ONVIF Integration** | High in commercial security cameras. | Low (Standard feature) | Medium. Requires specialized networking libraries. | **High.** ONVIF manages device discovery and configuration, working in tandem with RTSP to request specific video profiles. |
| **CCTV / IP Cameras** | Deployed at most major intersections. | High (Hardware procurement) | Hard. Requires physical mounting, networking, and power infrastructure. | **High (for Production).** Cameras must support dual-stream encoding: a low-resolution sub-stream for tracking and a high-resolution main stream for evidence cropping. |
| **Highway Cameras** | Operated by national authorities (e.g., NHAI). | Free (if access granted) | Hard. Bureaucratic hurdles to obtain network credentials. | **Moderate.** Excellent for scaling, but acquiring API access for a prototype delays development. |
| **Recorded Simulated Streams** | Infinite via prior datasets. | Free | Very Easy. | **Exceptional (for MVP).** Using software like VLC to loop an MP4 video to rtsp://localhost:\[span\_4\](start\_span)\[span\_4\](end\_span)\[span\_9\](start\_span)\[span\_9\](end\_span)8554/live simulates network packet ingestion flawlessly, bypassing municipal bureaucracy during the 4-8 week development phase. |

## **Deployment Target Analysis and Hardware Sizing**

The physical location of the compute hardware—whether centralized in a cloud environment or decentralized at the edge—dictates the inference frameworks (e.g., TensorRT) and the financial viability of the system.

| Deployment Target | Expected FPS | Hardware Requirements | Memory Requirements | Estimated Cost | Deployment Suitability |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Laptop Prototype** | 10–20 FPS | Intel i7 / AMD Ryzen 7, NVIDIA RTX 3060 / 4060 GPU. | 16GB System RAM, 6GB VRAM. | \~$1,000 \- $1,500 | Suitable exclusively for the 4-8 week development phase. Not viable for 24/7 continuous operation due to thermal throttling. |
| **Jetson Orin Nano** | 15–20 FPS (Single Stream) | NVIDIA Ampere GPU (1024 CUDA Cores), 6-core ARM CPU. | 8GB LPDDR5. | \~$250 \- $300 | Highly suitable for single-camera edge deployments where budget is severely constrained. Capable of up to 40 TOPS. |
| **Jetson Orin NX** | 30+ FPS (Multi-Stream) | NVIDIA Ampere GPU (1024 CUDA Cores, 32 Tensor Cores). | 8GB or 16GB LPDDR5. | \~$600 \- $800 | **Recommended Edge Target.** Provides up to 100 TOPS within a 10W-25W envelope. The expanded memory bandwidth is crucial for buffering high-resolution frames while running RT-DETR and PaddleOCR concurrently. |
| **Local Command Server** | 100+ FPS (Massive Multi-Stream) | Dual Intel Xeon / AMD EPYC, NVIDIA L4 or RTX 4000 Ada GPUs. | 64GB+ System RAM, 24GB+ VRAM per GPU. | $5,000 \- $10,000+ | **Recommended Centralized Target.** Processing feeds via fiber optic links at a central command center avoids placing delicate edge compute in harsh environmental conditions. |
| **Cloud Deployment** | Variable | AWS EC2 (g4dn/g5 instances), Azure NVv4. | Variable. | $0.50 \- $1.50 per hour / continuous. | Not Recommended. Streaming 24/7 high-definition video from municipal intersections to the cloud incurs catastrophic outbound bandwidth costs. |

## **Evidence Generation and Human-in-the-Loop Workflow**

A traffic violation system is entirely futile if it cannot produce legally defensible, structured evidence. Deep learning bounding boxes are merely internal variables; they must be packaged securely for judicial review.  
The system must maintain a rolling circular buffer in memory, capturing the last 10 seconds of high-resolution video. When the deterministic Rule Engine or the AI classifier triggers a violation event, the pipeline extracts the specific frame representing the apex of the violation. A localized crop of the vehicle and a tighter crop of the license plate are generated. The license plate crop is routed to the PaddleOCR engine for string extraction.  
A JSON payload is systematically assembled, containing the Unix Timestamp, intersection GPS Coordinates, the OCR-extracted Vehicle Number, the Violation Type, and the model's Confidence Scores. Crucially, the mathematical vectors (such as the virtual stop line or the directional trajectory path) and bounding boxes are graphically superimposed onto the high-resolution evidentiary frame. This composite image, alongside a 5-second video snippet exported from the rolling buffer, provides undeniable contextual proof of the offense.  
This payload is inserted into the PostgreSQL database, which feeds the React-based Command and Control dashboard. Here, the Human-in-the-Loop (HITL) workflow begins. An operator reviews the annotated image and video snippet, verifies the OCR string against the visual evidence, and executes an "Approve" or "Reject" command. Upon approval, the payload is formatted via a REST API to interface with national vehicular databases, such as the Indian VAHAN (vehicle registration) and SARATHI (driving licenses) systems, facilitating the automated generation and dispatch of an e-challan to the registered owner.

## **Novelty Opportunities for Real-World Superiority**

To elevate the prototype beyond standard academic implementations, practical innovations must be integrated to enhance explainability and reduce municipal labor.

* **Explainable Spatial Overlays:** Rather than presenting a black-box AI conclusion, the system must superimpose its mathematical reasoning onto the evidentiary image. By visibly displaying the trajectory vector of the vehicle intersecting the virtual stop-line polygon, the system provides mathematical proof, drastically reducing disputes from offenders.  
* **Dynamic ROI Auto-Calibration:** Hardcoded pixel coordinates for virtual loops fail catastrophically if a camera vibrates or shifts due to wind. The system implements an automated calibration routine utilizing edge detection and Hough Line Transforms to periodically identify the physical road markings and dynamically adjust the virtual ROI polygons to compensate for camera drift.  
* **Confidence-Based Triage Routing:** To maximize scalability, the HITL queue implements dynamic threshold routing. Violations exhibiting greater than 95% OCR confidence and unbroken tracking stability are automatically approved and dispatched. Only borderline cases (e.g., 70-95% confidence) are routed to human operators, saving thousands of hours of manual review.

## **Final Deliverable: Blueprint and Execution Roadmap**

### **System Architecture Data Flow**

1. RTSP Camera Stream continuously feeds the system.  
2. Frame Decoder (OpenCV/FFmpeg) decodes the stream, maintaining a high-resolution rolling buffer and passing low-resolution frames to the perception layer.  
3. Batch TensorRT Inference (RT-DETR/YOLOv10) executes object detection.  
4. ByteTrack Association assigns stable IDs to detected objects across frames.  
5. Rule Engine applies spatial geometry (Polygons/Vectors) to the tracked trajectories.  
6. \[Violation Triggered\] The system extracts the high-resolution frame and crops the license plate.  
7. PaddleOCR processes the crop to extract the registration string.  
8. Evidence Packager compiles the JSON payload, overlays graphics on the image, and cuts the video snippet.  
9. PostgreSQL Database stores the relational data.  
10. FastAPI Backend serves the data to the React/Streamlit Dashboard for HITL review.

### **Module-Wise Implementation Plan**

| Module | Core Responsibilities | Key Technologies |
| :---- | :---- | :---- |
| **Ingestion Module** | Connect to RTSP, handle frame dropping, manage circular memory buffers. | OpenCV, FFmpeg, Python collections.deque |
| **Perception Module** | TensorRT optimized execution of the object detection model. | RT-DETR/YOLOv10, PyTorch, NVIDIA TensorRT |
| **Tracking Module** | Kinematic association of bounding boxes. | ByteTrack, Kalman Filters, Hungarian Algorithm |
| **Logic Module** | Ray-casting, polygon intersection, coordinate geometry. | Shapely, NumPy |
| **Extraction Module** | Image cropping, image enhancement, text recognition. | OpenCV (CLAHE), PaddleOCR |
| **Storage & API Module** | Relational mapping, async request handling. | PostgreSQL, SQLAlchemy, FastAPI |

### **Development Roadmap**

**Minimum Viable Product (MVP) Phase (4–8 Weeks):**

* *Weeks 1-2:* Establish local infrastructure. Set up an RTSP loopback server utilizing pre-recorded traffic footage from the IDD dataset. Construct the FastAPI backend and define the PostgreSQL schema.  
* *Weeks 3-4:* Integrate the Apache 2.0 licensed RT-DETR model, fine-tuning it on a subset of DriveIndia and IDD data. Implement the ByteTrack algorithm and calibrate the Kalman filters to ensure ID stability.  
* *Weeks 5-6:* Develop the deterministic Rule Engine. Code the mathematical logic for virtual stop-lines and directional vectors for wrong-side driving. Integrate PaddleOCR to execute strictly upon violation triggers.  
* *Weeks 7-8:* Build the Streamlit dashboard for rapid prototyping of the HITL queue. Finalize the evidence generation workflow, ensuring synchronized video snipping and graphical overlays. Conduct rigorous stress testing to identify and patch memory leaks in the video buffer.

**Production-Scale Phase:**

* Port the core inference and tracking pipeline from Python to C++ utilizing NVIDIA DeepStream to maximize frame throughput on Jetson Orin NX edge devices.  
* Implement ONVIF device discovery protocols for seamless, automated onboarding of municipal IP cameras.  
* Initiate targeted data collection from the exact mounting perspectives of the deployment intersections to retrain models, mitigating accuracy drops caused by dense traffic occlusion.

### **Risk Analysis and Component Feasibility**

**Highly Achievable Components:** Wrong-side driving, stop-line violations, and red-light violations are highly deterministic and achievable due to their reliance on mature tracking algorithms and spatial geometry rather than visual classification. Vehicle detection and tracking are commoditized technologies backed by robust datasets like DriveIndia.  
**Difficult Components:** Seatbelt and mobile phone detection are fraught with failure points due to the lack of specialized public datasets reflecting diverse vehicle interiors, combined with the extreme physical constraints of windshield glare, window tinting, and poor cabin illumination. Attempting to deploy these without specialized cross-polarized and infrared camera hardware will result in an unacceptable false-positive rate, rendering the automated system legally untenable. These components must remain strictly out of scope for the MVP. Furthermore, deploying models with copyleft licenses (AGPL-3.0) introduces severe commercial liability; strict adherence to Apache 2.0 architectures (RT-DETR, PaddleOCR) mitigates this risk entirely.

#### **Works cited**

1\. e-CAM200\_CUONX \- 20MP (5K) AR2020 High Resolution Camera for NVIDIA® Jetson Orin NX / Orin Nano \- e-con Systems, https://www.e-consystems.com/nvidia-cameras/jetson-orin-nx-cameras/20mp-ar2020-high-resolution-camera.asp 2\. intelligent traffic management system (itms) \- Videonetics, https://www.videonetics.com/media/datasheet/new/ITMS.pdf 3\. WO2004111971A2 \- Automated traffic violation monitoring and reporting system with combined video and still-image data \- Google Patents, https://patents.google.com/patent/WO2004111971A2/en 4\. Orin Nano Vs Orin NX: Choosing The Right Jetson Module \- POS Virtual Research Quarterly, https://posvirtual.fapam.edu.br/fapam-news/orin-nano-vs-orin-nx-choosing-the-right-jetson-module-1767648565 5\. Edge AI Infrastructure: Deploying GPUs Closer to Data Sources \- Introl, https://introl.com/blog/edge-ai-infrastructure-deploying-gpus-closer-data-sources 6\. NVIDIA JetPack 6.2 Brings Super Mode to NVIDIA Jetson Orin Nano and Jetson Orin NX Modules | NVIDIA Technical Blog \- NVIDIA Developer, https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/ 7\. DriveIndia: An Object Detection Dataset for Diverse Indian Traffic Scenes \- arXiv, https://arxiv.org/html/2507.19912v4 8\. A Real-Time Illegal Parking Detection System with Automatic License Plate Number Recognition \- kyushu, https://catalog.lib.kyushu-u.ac.jp/opac\_download\_md/7395501/2025\_p0079.pdf 9\. A Deep Learning-Based System for Automatic License Plate Recognition Using YOLOv12 and PaddleOCR \- MDPI, https://www.mdpi.com/2076-3417/15/14/7833 10\. NATIONAL INFORMATICS CENTRE \- S3waas, https://cdnbbsr.s3waas.gov.in/s3584b98aac2dddf59ee2cf19ca4ccb75e/uploads/2021/09/2025072156.pdf 11\. Parivahan Sewa, mParivahan App, and How to Check Vehicle Details Online \- ClearTax, https://cleartax.in/s/parivahan-sewa-portal 12\. Real-time AI traffic violation detection using CNN \+ OpenCV — 89% accuracy, 15 FPS, pilot selected for city deployment \- GitHub, https://github.com/shgfx10/traffic-violation-detection 13\. Gurugram: GMDA to boost effective traffic management & road safety \- ET Infra, https://infra.economictimes.indiatimes.com/news/roads-highways/gurugram-gmda-to-boost-effective-traffic-management-road-safety/118739138 14\. YOLOv8 Model License and Pricing \- Roboflow, https://roboflow.com/model-licenses/yolov8 15\. Apache License 2.0 \- lyuwenyu/RT-DETR \- GitHub, https://github.com/lyuwenyu/RT-DETR/blob/main/LICENSE