#!/usr/bin/env bash
# Exit immediately if a command fails
set -o errexit

# Update package lists
apt-get update

# Install system dependencies
apt-get install -y tesseract-ocr gfortran

# Upgrade pip, setuptools, wheel
pip install --upgrade pip setuptools wheel

# Install Python dependencies
pip install -r requirements.txt
