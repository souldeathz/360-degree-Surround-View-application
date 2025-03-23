# Jetson AGX Xavier – Remote to Jetson AGX Xavier via SSH (for Docker + ROS 2 Humble)

```markdown
This guide explains how to connect from **Visual Studio Code on Windows** to your **Jetson AGX Xavier** running Ubuntu via **SSH**, typically over a **local network**.

## Prerequisites

- Jetson AGX Xavier already flashed and running Ubuntu (e.g., JetPack 5.x).
- Docker and ROS 2 Humble installed.
- Connected to the same network as your Windows PC.
- VS Code with Remote SSH extension installed.
- Your Jetson user account and password.
```


## Step-by-Step Instructions

### 🔧 Step 1: Enable SSH on Jetson AGX Xavier

SSH is usually enabled by default on Jetson. To verify:

```bash
sudo systemctl status ssh
```

If it's not active, enable and start it:

```bash
sudo systemctl enable ssh
sudo systemctl start ssh
```

---

### 📡 Step 2: Get Jetson's Local IP Address

On Jetson terminal:

```bash
ip a
```

Look for something like `192.168.x.x` under `eth0` or `wlan0`.

---

### 🧩 Step 3: Install Remote SSH Extension in VS Code

On your Windows machine:

1. Open **Visual Studio Code**
2. Go to Extensions (`Ctrl+Shift+X`)
3. Search and install **Remote - SSH** by Microsoft

---

### 🔐 Step 4: Add SSH Host to VS Code

1. Press `F1` or `Ctrl+Shift+P` to open command palette
2. Type and select: `Remote-SSH: Add New SSH Host...`
3. Enter SSH string:
   ```
   ssh <username>@<Jetson_IP>
   ```
   Example:
   ```
   ssh nvidia@192.168.1.100
   ```

4. Choose the config file (usually `C:\Users\<YourName>\.ssh\config`)
5. Save and close

---

### 📂 Step 5: Connect to Jetson via VS Code

1. Press `F1` or `Ctrl+Shift+P` again
2. Select `Remote-SSH: Connect to Host...`
3. Choose your Jetson entry
4. Enter password when prompted

You are now remotely connected to your Jetson AGX Xavier via SSH!

---

### 🐳 Step 6: Work with Docker + ROS 2

Once inside the Jetson environment:

- Use `docker ps`, `docker exec`, or `docker run` to interact with ROS 2 containers.
- You can also install the **Remote - Containers** extension in VS Code to attach directly to Docker containers for development.

---

## ✅ Tips

- Ensure port `22` is not blocked by firewall.
- You can set up **passwordless login** using SSH key for convenience (`ssh-keygen` + `ssh-copy-id`).
- For smoother file browsing and Docker integration, use extensions:
  - **Remote - Containers**
  - **Docker**

---

