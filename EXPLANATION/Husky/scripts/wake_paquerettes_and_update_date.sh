#!/bin/bash

# Get the current date and time from the host
current_date=$(date '+%Y%m%d %H:%M:%S')

# Remote machine credentials
REMOTE_USER="husky"
REMOTE_HOST="11.0.0.11"

# Prompt for the remote password (same for both SSH and sudo)
read -sp "Enter password for $REMOTE_USER@$REMOTE_HOST: " REMOTE_PASSWORD
echo

# Use sshpass to connect and pass the password to sudo
sshpass -p "$REMOTE_PASSWORD" ssh -X $REMOTE_USER@$REMOTE_HOST "echo $REMOTE_PASSWORD | sudo -S date -s \"$current_date\""
