#!/bin/bash

# Get the current year, month name, and day
YEAR=$(date +"%Y")
MONTH_NAME=$(date +"%B" | tr '[:upper:]' '[:lower:]')
DAY=$(date +"%d")

# Create the directory structure
mkdir -p "${YEAR}_husky_rosbags/${MONTH_NAME}_rosbags/${DAY}_rosbags"

# Output the created directory for confirmation
echo "Created directory structure: ${YEAR}_husky_rosbags/${MONTH_NAME}_rosbags/${DAY}_rosbags"
