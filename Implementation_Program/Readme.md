# Jetson AGX Xavier – Implementation Guide

> ⚠️ This guide assumes that all hardware and software preparations are already complete.  

---

## ✅ Prerequisites

Before proceeding with the installation, make sure you have:

- A Jetson AGX Xavier Developer Kit
- A host machine running **Ubuntu 20.04**
- A USB Type-C cable
- (Optional) M.2 SSD and PCIe USB 3.0 card installed
- Internet connection for downloading required software
- 4 Bird's Eye View cameras with proper mounting and USB connections 

📝 Also, make sure you've already followed:

- [Jetson AGX Xavier – Setup and Flashing Guide](./docs/Jetson%20AGX%20Xavier%20–%20Setup%20and%20Flashing%20Guide.md)  
  - Covers flashing the Xavier with JetPack and installing the SSD.
- [Jetson AGX Xavier – Docker and ROS 2 Humble Installation Guide](./docs/Jetson%20AGX%20Xavier%20–%20Docker%20Humble.md)  
  - Guides you through Docker installation and running ROS 2 Humble inside a container.
- [Jetson AGX Xavier – Remote Terminal Guide](./docs/Jetson%20AGX%20Xavier%20-%20Remote%20Terminal%20Guide.md)  
  - Explains how to use Visual Studio Code with Remote SSH to connect to Jetson from Windows.

---

## 🚀 Ready to Begin?

Once the Jetson is flashed, ROS 2 is running in Docker, and SSD is mounted:
1. Connect to the Jetson via SSH or VS Code Remote.
2. Navigate to your mounted SSD directory:
   ```bash
   cd /mnt/ssd/BFV_project
   ```
3. Launch your ROS 2 container:
   ```bash
   ./run_ros2_bfv.sh
   ```
4. Inside the container, navigate to the workspace:
   ```bash
   cd /root/code/ros2_ws
   colcon build
   source install/setup.bash
   ros2 run your_python_package your_node.py
   ```

---

## 📂 Project Implementation Structure

```
project-root/
│
├── README.md                                            <- Main project overview
├── docs/
│   ├── Jetson AGX Xavier – Setup and Flashing Guide.md  <- Flashing Jetson and SSD upgrade
│   ├── Jetson AGX Xavier – Docker Humble.md             <- Install Docker & ROS 2 Humble with container
│   ├── Jetson AGX Xavier – Remote Terminal Guide.md     <- VS Code Remote SSH setup for Jetson
│   └── jetson_agx_xavier_developer_kit_user_guide.pdf   <- Official NVIDIA documentation (optional)
│
├── Reference_Images/
│   ├── jetson_agx_xavier_1.jpg                          <- SSD and PCIe card install reference
│   ├── jetson_agx_xavier_2.jpg
│   └── ...
│
└── ros2_ws/                                             <- Your ROS 2 workspace
    ├── src/
    │   └── your-python-package/                         <- Place your ROS 2 nodes/packages here
    ├── install/                                         <- Auto-generated after build
    └── run_ros2_bfv.sh                                  <- Script to launch container with mounted code
```

---