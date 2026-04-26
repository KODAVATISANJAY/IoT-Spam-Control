# IoT Spam Control - System Architecture

## Overview
The IoT Spam Control system uses DSA and ML techniques to detect spam in IoT network traffic.

## Components

### Network Monitor
- Captures network packets in real-time
- Parses packet headers and payload
- Filters IoT-specific traffic

### Spam Detector (DSA)
- Pattern matching algorithms
- Bloom filters for fast lookups
- Hash-based detection

### ML Classifier
- Trained on spam traffic patterns
- Anomaly detection models
- Continuous learning pipeline

### Visualization Dashboard
- Real-time traffic monitoring
- Spam detection alerts
- Historical data charts

## Data Flow
1. Network Monitor captures packets
2. DSA module performs initial filtering
3. ML classifier scores suspicious traffic
4. Dashboard displays results in real-time
