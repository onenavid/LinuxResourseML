import threading
import time
import random
import datetime
import collections
import platform
import sys
import os
import csv
from tkinter import filedialog

# Third-party imports
try:
    import customtkinter as ctk
    import psutil
    import numpy as np
    from sklearn.ensemble import IsolationForest
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except ImportError as e:
    print("Missing required packages. Please install them using:")
    print("pip install customtkinter psutil scikit-learn matplotlib numpy pillow packaging")
    print(f"Error: {e}")
    sys.exit(1)

# --- Configuration ---
HISTORY_SIZE = 60  
UPDATE_INTERVAL_MS = 1000  
ANOMALY_TRAIN_SIZE = 60   
ML_RETRAIN_INTERVAL = 5 # Only retrain every 5 cycles for performance

# Configure CustomTkinter
ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"


class SystemData:
    """Data structure to hold current system metrics."""
    def __init__(self):
        self.cpu_percent = 0.0
        self.memory_percent = 0.0
        self.disk_percent = 0.0
        self.network_sent = 0.0  # MB/s
        self.network_recv = 0.0  # MB/s
        self.timestamp = datetime.datetime.now()

    def to_array(self):
        """Convert to array for ML model."""
        return [self.cpu_percent, self.memory_percent, self.disk_percent, self.network_sent + self.network_recv]


class MonitorThread(threading.Thread):
    """Background thread to collect system metrics."""
    def __init__(self, callback, interval=1.0):
        super().__init__()
        self.callback = callback
        self.interval = interval
        self.running = True
        self.prev_net_io = psutil.net_io_counters()
        self.prev_time = time.time()

    def run(self):
        while self.running:
            try:
                # CPU
                cpu = psutil.cpu_percent(interval=None)
                
                # Memory
                mem = psutil.virtual_memory().percent
                
                # Disk
                disk = psutil.disk_usage('/').percent
                
                # Network
                curr_net_io = psutil.net_io_counters()
                curr_time = time.time()
                time_delta = curr_time - self.prev_time if (curr_time - self.prev_time) > 0 else 1.0
                
                sent_mb = (curr_net_io.bytes_sent - self.prev_net_io.bytes_sent) / (1024 * 1024 * time_delta)
                recv_mb = (curr_net_io.bytes_recv - self.prev_net_io.bytes_recv) / (1024 * 1024 * time_delta)
                
                self.prev_net_io = curr_net_io
                self.prev_time = curr_time
                
                data = SystemData()
                data.cpu_percent = cpu
                data.memory_percent = mem
                data.disk_percent = disk
                data.network_sent = sent_mb
                data.network_recv = recv_mb
                
                self.callback(data)
                
                time.sleep(self.interval)
            except Exception as e:
                print(f"Error in monitor thread: {e}")
                time.sleep(1)

    def stop(self):
        self.running = False


class AnomalyDetector:
    """Manages the Isolation Forest model for anomaly detection."""
    def __init__(self):
        self.model = IsolationForest(contamination=0.05, random_state=42)
        self.data_history = []
        self.is_trained = False
        self.last_score = 0.0
        self.last_status = "Waiting..."
        self.counter = 0

    def add_data(self, data_point):
        self.data_history.append(data_point)
        if len(self.data_history) > 500:
            self.data_history.pop(0)

    def train_and_predict(self, current_data_point):
        self.counter += 1
        if len(self.data_history) < ANOMALY_TRAIN_SIZE:
            return "Collecting...", 0.0, 0.0

        X = np.array(self.data_history)
        
        try:
            # Optimize: Only retrain periodically
            if not self.is_trained or self.counter % ML_RETRAIN_INTERVAL == 0:
                self.model.fit(X)
                self.is_trained = True
            
            curr_X = np.array([current_data_point])
            prediction = self.model.predict(curr_X)[0]
            score = self.model.decision_function(curr_X)[0]
            confidence = min(abs(score) * 2, 1.0) if prediction == -1 else 0.0
            status = "Normal" if prediction == 1 else "ANOMALY"
            
            return status, score, confidence
        except Exception:
            return "Error", 0.0, 0.0


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("AI System Monitor & Anomaly Detector")
        self.geometry("1100x700")
        
        # Data handling
        self.monitor_thread = None
        self.anomaly_detector = AnomalyDetector()
        
        # Historical data for plots
        self.history_cpu = collections.deque(maxlen=HISTORY_SIZE)
        self.history_mem = collections.deque(maxlen=HISTORY_SIZE)
        self.history_timestamps = collections.deque(maxlen=HISTORY_SIZE)
        self.anomalies_indices = [] # Store timestamps where anomalies occurred

        self._setup_ui()
        
        # Start monitoring
        self.monitor_thread = MonitorThread(self._on_metrics_update, interval=1.0)
        self.monitor_thread.start()

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_ui(self):
        # Grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2) # Wider for charts
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar (Controls & Info) ---
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1)

        # Title
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Resource Monitor", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Metrics Labels
        self.lbl_cpu = self._create_metric_label(1, "CPU Usage")
        self.lbl_mem = self._create_metric_label(2, "Memory Usage")
        self.lbl_disk = self._create_metric_label(3, "Disk Usage")
        self.lbl_net = self._create_metric_label(4, "Network I/O")

        # ML Status
        self.ml_frame = ctk.CTkFrame(self.sidebar_frame)
        self.ml_frame.grid(row=5, column=0, padx=10, pady=20, sticky="ew")
        ctk.CTkLabel(self.ml_frame, text="ML Analysis", font=ctk.CTkFont(weight="bold")).pack(pady=5)
        self.lbl_ml_status = ctk.CTkLabel(self.ml_frame, text="Status: Initializing...", text_color="gray")
        self.lbl_ml_status.pack(pady=2)
        self.lbl_ml_score = ctk.CTkLabel(self.ml_frame, text="Score: --")
        self.lbl_ml_score.pack(pady=2)

        self.btn_export = ctk.CTkButton(self.sidebar_frame, text="Export Data (CSV)", command=self._export_data, fg_color="green")
        self.btn_export.grid(row=6, column=0, padx=20, pady=(20, 20), sticky="n")

        # --- Main Content Area ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_rowconfigure(0, weight=2) # Charts
        self.main_frame.grid_rowconfigure(1, weight=1) # Logs/Processes
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Charts Area
        self.chart_frame = ctk.CTkFrame(self.main_frame)
        self.chart_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 20))
        
        self._init_matplotlib()

        # Bottom Area: Top Processes & Alert Log
        self.bottom_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.bottom_frame.grid(row=1, column=0, sticky="nsew")
        self.bottom_frame.grid_columnconfigure(0, weight=1)
        self.bottom_frame.grid_columnconfigure(1, weight=1)

        # Processes
        self.proc_frame = ctk.CTkFrame(self.bottom_frame)
        self.proc_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        ctk.CTkLabel(self.proc_frame, text="Top Processes (CPU)", font=ctk.CTkFont(weight="bold")).pack(pady=5)
        self.txt_processes = ctk.CTkTextbox(self.proc_frame, height=150, font=("Consolas", 12))
        self.txt_processes.pack(fill="both", expand=True, padx=5, pady=5)

        # Alert Log
        self.log_frame = ctk.CTkFrame(self.bottom_frame)
        self.log_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        ctk.CTkLabel(self.log_frame, text="Alert Log", font=ctk.CTkFont(weight="bold")).pack(pady=5)
        self.txt_log = ctk.CTkTextbox(self.log_frame, height=150, font=("Consolas", 12))
        self.txt_log.pack(fill="both", expand=True, padx=5, pady=5)

    def _create_metric_label(self, row, title):
        frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        frame.grid(row=row, column=0, padx=20, pady=5, sticky="ew")
        lbl_title = ctk.CTkLabel(frame, text=title, anchor="w", font=("Arial", 12))
        lbl_title.pack(side="top", fill="x")
        lbl_val = ctk.CTkLabel(frame, text="--%", anchor="w", font=("Arial", 16, "bold"), text_color="cyan")
        lbl_val.pack(side="top", fill="x")
        return lbl_val

    def _init_matplotlib(self):
        # Create figure
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.fig.patch.set_facecolor('#2b2b2b') # Dark background match
        
        # Subplot
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#2b2b2b')
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        self.ax.set_title("Real-time System Metrics", color='white')
        self.ax.grid(True, color='#444444')
        
        # Lines
        self.line_cpu, = self.ax.plot([], [], label='CPU %', color="#1eff00", linewidth=1.5)
        self.line_mem, = self.ax.plot([], [], label='Memory %', color="#0400ff", linewidth=1.5)
        
        self.ax.legend(facecolor='#2b2b2b', labelcolor='white')

        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _on_metrics_update(self, data):
        # Use after() to schedule GUI updates on the main thread
        self.after(0, self._update_gui, data)

    def _update_gui(self, data):
        # Update Labels
        self.lbl_cpu.configure(text=f"{data.cpu_percent:.1f}%")
        self.lbl_mem.configure(text=f"{data.memory_percent:.1f}%")
        self.lbl_disk.configure(text=f"{data.disk_percent:.1f}%")
        self.lbl_net.configure(text=f"↑{data.network_sent:.1f} ↓{data.network_recv:.1f} MB/s")

        # ML Detection
        self.anomaly_detector.add_data(data.to_array())
        status, score, conf = self.anomaly_detector.train_and_predict(data.to_array())
        
        self.lbl_ml_status.configure(text=status)
        self.lbl_ml_score.configure(text=f"Score: {score:.2f}")
        
        if status != "Normal" and status != "Collecting Data":
            self.lbl_ml_status.configure(text_color="#f13636") # Red
            self._log_alert(f"ANOMALY: Score={score:.2f} Conf={conf:.2f}")
        elif status == "Normal":
            self.lbl_ml_status.configure(text_color="#41f158") # Green
        else:
             self.lbl_ml_status.configure(text_color="gray")

        # Update History & Charts
        self.history_cpu.append(data.cpu_percent)
        self.history_mem.append(data.memory_percent)
        self.history_timestamps.append(data.timestamp)
        # Simple integer X-axis for rolling window
        x_data = range(len(self.history_cpu))
        
        self.line_cpu.set_data(x_data, self.history_cpu)
        self.line_mem.set_data(x_data, self.history_mem)
        
        self.ax.set_xlim(0, max(HISTORY_SIZE, len(self.history_cpu)))
        self.ax.set_ylim(0, 100)
        self.canvas.draw()

        # Top Processes
        self._update_top_processes()

    def _update_top_processes(self):
        try:
            # Get top 5 by CPU
            procs = sorted(psutil.process_iter(['name', 'cpu_percent']), 
                           key=lambda p: p.info['cpu_percent'] or 0, 
                           reverse=True)[:5]
            
            text = "PID   Name                CPU%\n"
            text += "-"*35 + "\n"
            for p in procs:
                text += f"{str(p.pid).ljust(5)} {p.info['name'][:18].ljust(20)} {p.info['cpu_percent']}%\n"
            
            self.txt_processes.delete("1.0", "end")
            self.txt_processes.insert("1.0", text)
        except Exception:
            pass

    def _log_alert(self, message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.txt_log.insert("0.0", f"[{timestamp}] {message}\n")

    def _export_data(self):
        try:
            filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
            if not filename:
                return

            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Timestamp", "CPU %", "Memory %"])
                
                # Write history
                # Ensure all deques are same length (they should be)
                length = min(len(self.history_timestamps), len(self.history_cpu), len(self.history_mem))
                
                for i in range(length):
                    writer.writerow([
                        self.history_timestamps[i].strftime("%Y-%m-%d %H:%M:%S"),
                        self.history_cpu[i],
                        self.history_mem[i]
                    ])
            
            self._log_alert(f"Data exported to {os.path.basename(filename)}")
        except Exception as e:
            self._log_alert(f"Export failed: {str(e)}")

    def _on_close(self):
        if self.monitor_thread:
            self.monitor_thread.stop()
        self.destroy()
        sys.exit(0)

if __name__ == "__main__":
    try:
        app = App()
        app.mainloop()
    except Exception as e:
        print(f"Failed to start GUI: {e}")
        print("Note: This application requires a graphical desktop environment.")
