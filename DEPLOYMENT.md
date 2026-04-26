# IoT Spam Control - Deployment Guide

## Local Setup
1. Clone the repository
2. Install dependencies: pip install -r requirements.txt
3. Configure .env file with database credentials
4. Run: python main.py

## Cloud Deployment
- AWS EC2 for backend server
- AWS RDS for database
- AWS CloudWatch for monitoring

## Docker
```bash
docker build -t iot-spam-control .
docker run -p 8080:8080 iot-spam-control
```

## Environment Variables
- DATABASE_URL
- API_KEY
- SPAM_THRESHOLD
- LOG_LEVEL
