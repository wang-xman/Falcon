#!/bin/bash
# Run this script with sudo.
apt update
apt install -y build-essential
apt install -y bison flex libelf-dev curl dkms cmake systemd
apt install -y python3-pip
apt install -y python3-virtualenv
apt install python3.10-venv
