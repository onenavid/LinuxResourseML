# Linux Resource Monitoring & Anomaly Detection

**Author:** Naveed Hayat & Umer Farooq\
**Course:** BS Data Science, 5th Semester\
**Project:** Operating System Project

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Code Structure](#code-structure)
6. [Classes & Functions](#classes--functions)
7. [Flow of Program](#flow-of-program)
8. [Screenshots / UI](#screenshots--ui)
9. [References](#references)

---

## Project Overview

This project is a **real-time system monitoring tool** with an integrated **anomaly detection system** using machine learning (Isolation Forest). The application monitors CPU, memory, disk, and network usage, logs anomalies, and displays real-time plots and top CPU-consuming processes.

The GUI is built using **CustomTkinter** for a modern dark theme, and data visualization is done using **Matplotlib**. The ML model detects **anomalies in system behavior** and highlights them for the user, making this project useful for performance monitoring or learning ML applications in system analytics.

---

## Features

- Real-time monitoring of:
  - CPU usage
  - Memory usage
  - Disk usage
  - Network activity (sent/received MB/s)
- Real-time plotting of CPU & Memory usage
- Anomaly detection using **Isolation Forest**
- Top CPU-consuming processes display
- Alerts log for anomalies
- Export history data to **CSV**
- Modern dark-themed GUI using **CustomTkinter**

---

## Installation

1. Clone the repository or copy the source code.
2. Install Python 3.10+ if not already installed.
3. Install required packages:

```bash
pip install customtkinter psutil numpy scikit-learn matplotlib pillow packaging
```

4. Run the program:

```bash
python app.py
```

> Note: This program requires a graphical desktop environment.

---

## Usage

1. Launch the application.
2. The sidebar displays **real-time CPU, Memory, Disk, and Network metrics**.
3. The main panel shows:
   - Real-time plots of CPU & Memory usage.
   - Top processes by CPU usage.
   - Alerts log for anomalies.
4. Click **“Export Data (CSV)”** to save historical metrics.
5. Close the window to **stop monitoring safely**.

---

## Code Structure

```text
project/
│
├── app.py                # Main application file
├── README.md             # Project documentation
├── requirements.txt      # Optional: pip requirements
└── assets/               # Optional folder for UI assets
```

---

## Classes & Functions

### **SystemData**

- Stores **CPU, memory, disk, network, and timestamp**.
- `to_array()` → Converts metrics to a list for ML.

### **MonitorThread**

- Runs in a **background thread** to collect system metrics.
- Measures:
  - CPU % using `psutil.cpu_percent()`
  - Memory % using `psutil.virtual_memory().percent`
  - Disk % using `psutil.disk_usage('/').percent`
  - Network MB/s by calculating difference in bytes sent/received
- Calls a **callback function** to update the GUI.
- Stops safely using `stop()`.

### **AnomalyDetector**

- Uses **Isolation Forest** to detect anomalies.
- `add_data(data_point)` → Add new metrics to history.
- `train_and_predict(current_data_point)` → Trains periodically and predicts anomaly:
  - Returns `status` (Normal / Anomaly), `score`, `confidence`.

### **App (GUI)**

- Main GUI class built with **CustomTkinter**.
- Components:
  - **Sidebar**: Metrics, ML status, Export button
  - **Main panel**: Plots, top processes, alert log
- Updates:
  -  → Called by monitoring thread
  -  → Updates labels, charts, and processes
- CSV Export:
- Graceful shutdown

---

- **Data flow**:
  1. `MonitorThread` collects metrics.
  2. Calls GUI callback (`_on_metrics_update`).
  3. GUI updates labels and charts.
  4. ML model predicts anomalies.
  5. Alerts are logged, and top processes are updated.
  6. Data is stored in history for plotting and CSV export.

---

## References

- [CustomTkinter Documentation](https://github.com/TomSchimansky/CustomTkinter)
- [psutil Documentation](https://psutil.readthedocs.io/en/latest/)
- [Isolation Forest (scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

