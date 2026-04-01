#!/bin/bash

# Start base image services (SSH) in background
/services/start.sh &

# Wait a moment for services to start
sleep 2

# Run application command
python /services/main.py

# Wait for background processes
wait