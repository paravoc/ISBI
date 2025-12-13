# ISBI: Information System for Biometric Identification

Software for video capture, facial feature analysis and comparison, access control system management

## Project Overview
Scientific research project for implementation in institutional access control systems. The system performs real-time biometric identification from video streams with subsequent turnstile control.

## System Architecture

### Development Branches:
| Branch | Purpose | Technologies |
|--------|---------|--------------|
| **`testingOPENCV`** | Video capture from multiple cameras | OpenCV, FFmpeg, RTSP |
| **`testingCAMERA`** | Feature extraction and database operations | Dlib, FaceNet, SQLite/PostgreSQL |
| **`testingBASTION`** | Turnstile control (Bastion X architecture) | Modbus, REST API, MQTT |
| **`main`** | Main stable version | Integration of all components |

