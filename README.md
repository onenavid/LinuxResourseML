# AI System Monitor & Anomaly Detector - Enhanced with Paging Analysis

**Author:** Naveed Hayat & Umer Farooq  
**Course:** BS Data Science, 5th Semester  
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
8. [Memory Paging Metrics](#memory-paging-metrics)
9. [Anomaly Detection](#anomaly-detection)
10. [Screenshots / UI](#screenshots--ui)
11. [References](#references)

---

## Project Overview

This project is an **advanced real-time system monitoring tool** with integrated **machine learning-based anomaly detection** using Isolation Forest. The application provides comprehensive monitoring of system resources including CPU, memory, disk, network, and **advanced memory paging metrics** such as swap usage, page faults, and swap I/O rates.

The GUI is built using **CustomTkinter** for a modern dark theme, with dual-chart data visualization powered by **Matplotlib**. The ML model intelligently detects **anomalies in system behavior** and identifies which specific metrics triggered the alert, making this tool essential for performance monitoring, system diagnostics, and learning ML applications in system analytics.

---

## Features

### Real-time Monitoring
- **CPU usage** - Percentage utilization
- **Memory usage** - RAM consumption percentage
- **Disk usage** - Storage utilization
- **Network activity** - Upload/download rates in MB/s
- **Swap usage** - Virtual memory percentage
- **Page faults** - Rate of page faults per second
- **Swap I/O rates** - Swap in/out activity (Linux)

### Visualization
- **Dual real-time charts:**
  - Chart 1: CPU & Memory usage trends
  - Chart 2: Memory paging metrics (Swap % & Page Faults/s)
- **Color-coded alerts** for critical thresholds
- **Auto-scaling axes** for optimal visualization

### Anomaly Detection
- **Isolation Forest ML model** with adaptive training
- **Intelligent anomaly identification** - identifies which specific metrics triggered alerts
- **Memory pressure detection** - specialized alerts for paging-related anomalies
- **Confidence scoring** for anomaly predictions
- **Historical data analysis** with 500-point rolling window

### Additional Features
- **Top processes display** - Shows top 5 CPU-consuming processes
- **Alert logging system** - Timestamped anomaly log
- **CSV data export** - Export historical metrics for analysis
- **Cross-platform support** - Works on Linux and Windows
- **Modern dark-themed GUI** using CustomTkinter
- **Thread-safe monitoring** with background data collection

---

## Installation

### Prerequisites
- Python 3.10 or higher
- Graphical desktop environment
- Administrator/root privileges (recommended for full metrics access)

### Steps

1. Clone the repository or download the source code:
```bash
git clone <repository-url>
cd <project-directory>
```

2. Install required packages:
```bash
pip install customtkinter psutil numpy scikit-learn matplotlib pillow packaging
```

3. Run the application:
```bash
python LRMv.py
```

> **Note:** On Linux, some paging metrics may require elevated privileges. Run with `sudo` if needed:
> ```bash
> sudo python LRMv.py
> ```

---

## Usage

### Getting Started

1. **Launch the application** - Run `python LRMv.py`
2. **Monitor real-time metrics** - The sidebar displays live system statistics
3. **Observe trends** - Two charts show historical data:
   - Left chart: CPU & Memory usage
   - Right chart: Swap usage & Page faults
4. **Watch for anomalies** - The ML Analysis section will alert you to unusual behavior
5. **Check processes** - Bottom left shows top CPU-consuming processes
6. **Review alerts** - Bottom right displays the anomaly alert log

### Understanding the Interface

#### Sidebar Metrics
- **CPU Usage** - Current processor utilization
- **Memory Usage** - RAM consumption
- **Disk Usage** - Storage utilization
- **Network I/O** - ↑ Upload / ↓ Download speeds
- **Swap Usage** - Virtual memory (color-coded: cyan=normal, orange=warning, red=critical)
- **Page Faults/s** - Memory paging activity (displayed as faults/s or k/s for large values)

#### ML Analysis Panel
- **Status** - "Collecting...", "Normal", or "ANOMALY"
- **Score** - Anomaly score from the ML model
- **Triggers** - Specific metrics that caused the anomaly (CPU, Memory, Disk, Network, Swap, Page Faults, Swap I/O)

#### Special Alerts
- **Memory Pressure** - Triggered when high paging activity is detected (Swap or Page Faults anomalies)

### Exporting Data

Click **"Export Data (CSV)"** to save historical metrics including:
- Timestamp
- CPU %
- Memory %
- Swap %
- Page Faults/s

---

## Code Structure

```text
project/
│
├── LRMv.py              # Main application file
├── README.md            # Project documentation (this file)
└── requirements.txt     # Optional: pip requirements
```

### Configuration Constants

```python
HISTORY_SIZE = 60              # Data points to display in charts
UPDATE_INTERVAL_MS = 1000      # GUI update interval (ms)
ANOMALY_TRAIN_SIZE = 60        # Minimum data points before ML training
ML_RETRAIN_INTERVAL = 5        # Retrain model every N updates
```

---

## Classes & Functions

### **SystemData**
Data structure holding current system metrics.

**Attributes:**
- `cpu_percent` - CPU usage percentage
- `memory_percent` - Memory usage percentage
- `disk_percent` - Disk usage percentage
- `network_sent` - Upload speed (MB/s)
- `network_recv` - Download speed (MB/s)
- `swap_percent` - Swap usage percentage
- `page_faults_per_sec` - Page fault rate
- `swap_in_rate` - Swap-in rate (MB/s on Linux)
- `swap_out_rate` - Swap-out rate (MB/s on Linux)
- `timestamp` - Collection time

**Methods:**
- `to_array()` → Converts metrics to numpy array for ML model (7 features)

---

### **MonitorThread**
Background thread for continuous system metrics collection.

**Key Features:**
- Non-blocking background execution
- Platform-aware metric collection (Linux vs Windows)
- Rate calculation for network and swap I/O
- Aggregate page fault monitoring across all processes

**Methods:**
- `run()` - Main monitoring loop
- `stop()` - Graceful thread termination

**Metrics Collection:**
- **CPU:** `psutil.cpu_percent(interval=None)`
- **Memory:** `psutil.virtual_memory().percent`
- **Disk:** `psutil.disk_usage('/').percent`
- **Network:** Calculates MB/s from byte delta over time
- **Swap:** `psutil.swap_memory()` with platform-specific I/O rates
- **Page Faults:** Aggregates from all accessible processes

---

### **AnomalyDetector**
Machine learning engine using Isolation Forest for anomaly detection.

**Configuration:**
- Contamination rate: 5% (expects 5% of data to be anomalous)
- Random state: 42 (reproducible results)
- Rolling history: 500 data points maximum

**Methods:**
- `add_data(data_point)` - Adds metrics to training history
- `train_and_predict(current_data_point)` - Main detection method
  - Returns: status, score, confidence, list of anomaly triggers
  - Periodic retraining (every 5 updates after initial training)
  - Feature-level anomaly analysis using 2-sigma rule

**Anomaly Trigger Detection:**
Compares current values to recent averages (last 20 points) and flags features that deviate by more than 2 standard deviations.

---

### **App (Main GUI)**
CustomTkinter-based application window.

**Layout:**
- **Sidebar** (left): Metrics display, ML analysis, export button
- **Main area** (right): 
  - Top: Dual charts (CPU/Memory + Paging metrics)
  - Bottom: Process list and alert log

**Key Methods:**
- `_setup_ui()` - Initializes all GUI components
- `_init_matplotlib()` - Creates two matplotlib figures
- `_on_metrics_update(data)` - Thread-safe callback from MonitorThread
- `_update_gui(data)` - Updates all UI elements
- `_update_top_processes()` - Refreshes process list
- `_log_alert(message)` - Adds timestamped entry to alert log
- `_export_data()` - Saves metrics to CSV file
- `_on_close()` - Cleanup and safe shutdown

**Chart Configuration:**
- **Chart 1:** CPU (green) & Memory (blue) on shared Y-axis (0-100%)
- **Chart 2:** Swap (orange, left Y-axis) & Page Faults (magenta, right Y-axis with auto-scaling)

---

## Flow of Program

### Initialization
1. Application starts → `App.__init__()`
2. GUI components created → `_setup_ui()`, `_init_matplotlib()`
3. Monitoring thread spawned → `MonitorThread.start()`
4. Thread begins collecting metrics every 1 second

### Runtime Loop
1. **MonitorThread** collects system metrics
2. Calculates rates (network, swap I/O, page faults)
3. Creates `SystemData` object
4. Calls GUI callback → `_on_metrics_update(data)`
5. GUI thread-safely updates via `_update_gui(data)`:
   - Updates metric labels with color coding
   - Adds data to ML model → `AnomalyDetector.add_data()`
   - Runs anomaly prediction → `train_and_predict()`
   - Logs alerts if anomalies detected
   - Appends to history deques
   - Redraws both matplotlib charts
   - Updates top processes list
6. Loop repeats every second

### Anomaly Detection Flow
1. New data point added to 500-point rolling history
2. Every 5 updates, model retrains on historical data
3. Current data point evaluated against model
4. If anomalous:
   - Compare to recent average ± 2σ for each feature
   - Identify which metrics triggered the anomaly
   - Generate alert with specific triggers
   - Special memory pressure alert for swap/paging issues
5. Update ML status display with results

### Shutdown
1. User closes window → `_on_close()`
2. Monitoring thread stopped → `MonitorThread.stop()`
3. Application destroyed → `App.destroy()`
4. Clean exit → `sys.exit(0)`

---

## Memory Paging Metrics

### What is Memory Paging?

Memory paging is a memory management technique where the OS moves data between RAM and disk storage (swap space). Monitoring paging activity helps identify **memory pressure** - when the system lacks sufficient RAM and must rely on slower disk-based virtual memory.

### Monitored Paging Metrics

#### 1. **Swap Usage (%)**
- Percentage of swap space currently in use
- **Color coding:**
  - Cyan (< 20%): Normal
  - Orange (20-50%): Warning
  - Red (> 50%): Critical
- High swap usage indicates memory shortage

#### 2. **Page Faults per Second**
- Rate at which the system accesses memory pages not in RAM
- Aggregated across all processes
- High rates indicate:
  - Insufficient physical memory
  - Programs accessing more memory than available
  - Potential performance degradation

#### 3. **Swap I/O Rates** (Linux only)
- **Swap In:** Data moved from disk to RAM (MB/s)
- **Swap Out:** Data moved from RAM to disk (MB/s)
- Continuous swap I/O indicates **thrashing** - severe performance issue

### Why Monitor Paging?

- **Performance diagnosis** - High paging = slow system
- **Capacity planning** - Identify when more RAM is needed
- **Anomaly detection** - Sudden paging spikes indicate issues
- **Process optimization** - Identify memory-hungry applications

---

## Anomaly Detection

### Machine Learning Model

**Algorithm:** Isolation Forest (unsupervised learning)
- Designed for anomaly detection in multidimensional data
- Doesn't require labeled training data
- Isolates anomalies rather than profiling normal behavior

### How It Works

1. **Training Phase** (after collecting 60 data points):
   - Model learns normal system behavior patterns
   - Retrains every 5 updates to adapt to changing conditions

2. **Prediction Phase**:
   - Each new data point scored against learned patterns
   - Negative score → Anomaly detected
   - Positive score → Normal behavior

3. **Feature Analysis**:
   - Identifies which specific metrics are anomalous
   - Uses 2-sigma rule: values > 2 standard deviations from recent average

### Anomaly Types Detected

- **CPU spikes** - Sudden processor load
- **Memory pressure** - Unusual RAM consumption
- **Disk saturation** - Storage utilization spikes
- **Network anomalies** - Unusual traffic patterns
- **Swap thrashing** - Excessive virtual memory use
- **Page fault storms** - Memory paging issues
- **Combined anomalies** - Multiple simultaneous issues

### Special Alerts

**Memory Pressure Warning:**
Triggered when anomalies involve Swap or Page Faults metrics, indicating the system is struggling with memory management.

---

## Screenshots / UI

### Main Interface
- **Left sidebar:** Real-time metrics with color-coded values
- **Top-right:** Dual charts showing resource trends and paging activity
- **Bottom-right:** Split view with process list and alert log

### Visual Indicators
- **Green line:** CPU usage
- **Blue line:** Memory usage
- **Orange line:** Swap usage
- **Magenta line:** Page fault rate
- **Color-coded status:** Gray (collecting) → Green (normal) → Red (anomaly)

---

## References

### Libraries & Documentation
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) - Modern tkinter UI library
- [psutil](https://psutil.readthedocs.io/en/latest/) - Cross-platform system monitoring
- [scikit-learn: Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) - Anomaly detection algorithm
- [Matplotlib](https://matplotlib.org/stable/contents.html) - Data visualization
- [NumPy](https://numpy.org/doc/stable/) - Numerical computing

### Academic References
- Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. *IEEE International Conference on Data Mining*.
- Operating Systems: Three Easy Pieces - Memory Paging chapter
- Modern Operating Systems by Andrew Tanenbaum - Virtual Memory section

---

## License

This project is created for educational purposes as part of an Operating Systems course project.

---

## Contributing

**Students:** Naveed Hayat & Umer Farooq  
**Institution:** BS Data Science Program  
**Semester:** 5th  

For questions or improvements, please contact the authors.
