```
The MIT License (MIT)

Copyright (c) 2025 Matt Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
# Multi-Chip Multi-Modal Multi-Stream Inference

This repo is a step-by-step guide to a minimal working example (MWE)
for running livestream inference using multiple HAILO-8 chips.
The key feature of a multiple-chip accelerator is the possibility
to run different neural networks simultaneously.
All codes are tested on
[FALCON-H8 AI Accelerator](https://www.lannerinc.com/products/edge-ai-appliance/deep-learning-accelerators/falcon-h8) from
[Lanner Electronics](https://www.lannerinc.com/).

Modules in this repo are largely based on [Hailo Application Code Examples](https://github.com/hailo-ai/Hailo-Application-Code-Examples).
Modification was made to allow users to select a specific HAILO device
in a multip-chip evrionment.

## Hardware, system packages, and virtual environment

The host system is [Lanner LEC-2290H](https://www.lannerinc.com/products/edge-ai-appliance/deep-learning-inference-appliances/lec-2290h) with a
4-chip FALCON accelerator. It runs on [Ubuntu 22.04 LTS](https://releases.ubuntu.com/jammy/) which comes with Python 3.10.

Before installing HAILO driver and runtime, the following packages
are required
```
sudo apt update
sudo apt install -y build-essential
sudo apt install -y bison flex libelf-dev curl dkms cmake systemd
sudo apt install -y python3-pip
sudo apt install -y python3-virtualenv
sudo apt install python3.10-venv
```
You may also run the script `Scripts/install_prerequisite.sh`.

From [HAILO DevZone](https://hailo.ai/developer-zone/), download Hailo
PCIe driver and runtime. I usually use the Debian package. For this
MWE, I installed the version 4.21 for Python 3.10. So simply,
```
sudo dpkg -i hailort-pcie-driver_4.21.0_all.deb
sudo dpkg -i hailort_4.21.0_amd64.deb
```

Create a virtual environment, say
```
python3 -m venv ./venv_falcon
source ./venv_falcon/bin/activate
```
Again from [HAILO DevZone](https://hailo.ai/developer-zone/), download the
Hailo-RT Python package `hailort-4.21.0-cp310-cp310-linux_x86_64.whl` and
install it in the virtual environment
```
(venv_falcon)$ pip install hailort-4.21.0-cp310-cp310-linux_x86_64.whl
```
Please make sure that Hailo-RT Python package has the same version as
Hailo runtime. After this, install the following
```
argcomplete==3.6.2
contextlib2==21.6.0
future==1.0.0
loguru==0.7.3
netaddr==1.3.0
netifaces==0.11.0
numpy==1.26.4
opencv-python==4.11.0.86
pillow==11.2.1
```
You may also use the requirement file `Scripts/requirements_falcon_x86u22.txt`
for installation,
```
(venv_falcon)$ pip install -r requirements_falcon_x86u22.txt
```

## Run inference on livestream

On the FALCON-H8 I used for testing, there are 4 HAILO chips, indexed from 0 to 3. Inferencing using Hailo chip 1, camera 2, neural net `YOLOv7`, and a batch size 8,
```
(venv_falcon)$ python3 example.py --input_device camera --camera_index 2 --hailo_device 1 --network_path ./yolov7.hef
```
You may launch other pipelines using different camera indices, Hailo devices,
and neural networks.

## Run inference on video clip

Meanwhile, you may also launch inferecing pipelines using recorded video clips.
In this example, I used neurel net `YOLOv8m`, and a batch size 1. Here camera index is
actually irrelevant
```
(venv_falcon)$ python3 example.py --input_device walkers.mp4 --camera_index 0 --hailo_device 1 --network_path ./yolov8m.hef --batch_size 1
```