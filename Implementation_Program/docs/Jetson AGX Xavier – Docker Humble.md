# Jetson AGX Xavier – ROS 2 on Docker

## 📦 Setup Guide: Running ROS 2 (Humble) in Docker on Jetson AGX Xavier

---

### 🔧 Required Hardware & Software

- **Jetson AGX Xavier**
- **Python**: Version 3.8.3 to 3.10.0 (recommended for ROS 2 Humble)
- **Ubuntu**: JetPack-based Ubuntu OS (typically 20.04)
- **Docker**

---

## 1. 🐳 Install Docker on Jetson AGX Xavier

```bash
# Update package index
sudo apt update
sudo apt upgrade -y

# Install required dependencies
sudo apt install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker’s GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Add Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Update again
sudo apt update

# Install Docker Engine
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Check Docker version
docker --version
```

---

## 2. 🐳 Download ROS 2 Docker Image

```bash
docker pull ros:humble
```

---

## 3. 🐳 Using Docker with ROS 2

### ✅ Create a container with SSD mounted

```bash
docker run -it --name ros2_playground \
  -v /mnt/ssd/BFV_project:/root/code \
  ros:humble
```

### 🔁 Restart existing container

```bash
docker start -ai ros2_playground
```

### ❌ Remove container (if needed)

```bash
docker rm -f ros2_playground
```

---

## ⚙️ Initial Setup Inside the Container (One-time)

```bash
apt update && apt install curl gnupg lsb-release -y
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add -
sh -c 'echo "deb http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" > /etc/apt/sources.list.d/ros2-latest.list'
apt update
apt install ros-humble-demo-nodes-cpp python3-pip -y
```

> 🔑 Don’t forget to source ROS 2 before running any `ros2` commands:

```bash
source /opt/ros/humble/setup.bash
```

---

## 🐍 Running Python ROS Nodes

```bash
cd /root/code
python3 my_talker.py
# or
python3 my_listener.py
```

---

## ❗ Common Issues & Fixes

| Problem | Solution |
|--------|----------|
| `ros2: command not found` | Run `source /opt/ros/humble/setup.bash` |
| `python3: can't open file` | Ensure you're in the `/root/code` directory |
| Container disappears after closing | Don't use `--rm` flag with `docker run` |
| VS Code can't see mounted SSD | Use `File > Open Folder...`, then enter `/mnt/ssd/BFV_project` manually |

---

## 🧰 Handy Script: `run_ros2_bfv.sh`

```bash
#!/bin/bash
docker rm -f ros2_playground 2>/dev/null
docker run -it --name ros2_playground \
  -v /mnt/ssd/BFV_project:/root/code \
  ros:humble
```

---

## 🧠 Auto-Source ROS 2 on Container Start

To avoid running `source /opt/ros/humble/setup.bash` every time:

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

Verify:

```bash
cat ~/.bashrc
```

You should see:

```bash
source /opt/ros/humble/setup.bash
```

---

## 💾 Auto-Mount SSD on Boot

### 1. Find the UUID of your SSD

```bash
sudo blkid /dev/nvme0n1p1
```

Example output:

```
/dev/nvme0n1p1: UUID="4f11d2f9-11fd-4b53-b8f0-c0c2e638b97f" TYPE="ext4"
```

### 2. Edit the `fstab` file

```bash
sudo nano /etc/fstab
```

Add at the bottom:

```
UUID=4f11d2f9-11fd-4b53-b8f0-c0c2e638b97f  /mnt/ssd  ext4  defaults  0  2
```

### 3. Test it

```bash
sudo mount -a
```

If no errors occur, your SSD will auto-mount on boot.

---

