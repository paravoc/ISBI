# ISBI: Information System for Biometric Identification

Software for video capture, facial feature analysis and comparison, access control system management

## Project Overview
Scientific research project for implementation in institutional access control systems. The system performs real-time biometric identification from video streams with subsequent turnstile control.

## System Architecture

### Development Branches:
| Branch | Purpose | Technologies |
|--------|---------|--------------|
| **`testingOPENCV`** | Video capture from multiple cameras | OpenCV |
| **`testingCAMERA`** | Feature extraction and database operations | PostgreSQL, extension pgvector |
| **`testingBASTION`** | Turnstile control (Bastion X architecture) | REST API |
| **`testingSPUFING`** | Comparative analysis of methods and tools for detecting spoofing attacks | Different methods of protection against spoofing attacks |
| **`main`** | Main stable version | Integration of all components |


