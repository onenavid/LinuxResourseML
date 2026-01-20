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
ML_RETRAIN_INTERVAL = 5

# Configure CustomTkinter
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")


class SystemData:
    """Data structure to hold current system metrics."""
    def __init__(self):
        self.cpu_percent = 0.0
        self.memory_percent = 0.0
        self.disk_percent = 0.0
        self.network_sent = 0.0  # MB/s
        self.network_recv = 0.0  # MB/s
        self.swap_percent = 0.0
        self.page_faults_per_sec = 0.0  # Combined major + minor
        self.swap_in_rate = 0.0  # Pages/sec (Linux) or MB/s (Windows)
        self.swap_out_rate = 0.0
        self.timestamp = datetime.datetime.now()

    def to_array(self):
        """Convert to array for ML model - now includes paging metrics."""
        return [
            self.cpu_percent, 
            self.memory_percent, 
            self.disk_percent, 
            self.network_sent + self.network_recv,
            self.swap_percent,
            self.page_faults_per_sec,
            self.swap_in_rate + self.swap_out_rate
        ]


class MonitorThread(threading.Thread):
    """Background thread to collect system metrics."""
    def __init__(self, callback, interval=1.0):
        super().__init__()
        self.callback = callback
        self.interval = interval
        self.running = True
        self.prev_net_io = psutil.net_io_counters()
        self.prev_time = time.time()
        
        # For paging metrics
        self.prev_swap_stats = None
        self.is_linux = platform.system() == 'Linux'
        self.prev_page_faults = 0
        
        # Initialize swap stats
        try:
            if self.is_linux:
                self.prev_swap_stats = psutil.swap_memory()
        except:
            pass

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
                
                # PAGING METRICS
                swap_percent = 0.0
                page_faults_per_sec = 0.0
                swap_in_rate = 0.0
                swap_out_rate = 0.0
                
                # Swap usage
                try:
                    swap = psutil.swap_memory()
                    swap_percent = swap.percent
                    
                    # Swap in/out rates (Linux only)
                    if self.is_linux and self.prev_swap_stats:
                        if hasattr(swap, 'sin') and hasattr(swap, 'sout'):
                            # sin/sout are cumulative bytes, calculate rate
                            sin_delta = swap.sin - self.prev_swap_stats.sin
                            sout_delta = swap.sout - self.prev_swap_stats.sout
                            
                            swap_in_rate = sin_delta / (1024 * 1024 * time_delta)  # MB/s
                            swap_out_rate = sout_delta / (1024 * 1024 * time_delta)
                    
                    self.prev_swap_stats = swap
                except:
                    pass
                
                # Page faults (aggregate from all processes)
                try:
                    total_page_faults = 0
                    for proc in psutil.process_iter(['pid']):
                        try:
                            # Get page fault info
                            if hasattr(proc, 'memory_info'):
                                mem_info = proc.memory_info()
                                # Windows: pagefaults, Linux: page_faults (not always available)
                                if hasattr(mem_info, 'num_page_faults'):
                                    total_page_faults += mem_info.num_page_faults
                                elif hasattr(mem_info, 'pfaults'):
                                    total_page_faults += mem_info.pfaults
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
                    
                    # Calculate rate
                    if self.prev_page_faults > 0:
                        page_faults_per_sec = (total_page_faults - self.prev_page_faults) / time_delta
                    self.prev_page_faults = total_page_faults
                except:
                    pass
                
                data = SystemData()
                data.cpu_percent = cpu
                data.memory_percent = mem
                data.disk_percent = disk
                data.network_sent = sent_mb
                data.network_recv = recv_mb
                data.swap_percent = swap_percent
                data.page_faults_per_sec = max(0, page_faults_per_sec)  # Ensure non-negative
                data.swap_in_rate = max(0, swap_in_rate)
                data.swap_out_rate = max(0, swap_out_rate)
                
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
        self.anomaly_reasons = []  # Track which features triggered anomaly

    def add_data(self, data_point):
        self.data_history.append(data_point)
        if len(self.data_history) > 500:
            self.data_history.pop(0)

    def train_and_predict(self, current_data_point):
        self.counter += 1
        if len(self.data_history) < ANOMALY_TRAIN_SIZE:
            return "Collecting...", 0.0, 0.0, []

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
            
            # Identify anomaly reasons
            reasons = []
            if prediction == -1:
                feature_names = ["CPU", "Memory", "Disk", "Network", "Swap", "Page Faults", "Swap I/O"]
                # Compare to recent averages
                recent_avg = np.mean(X[-20:], axis=0) if len(X) >= 20 else np.mean(X, axis=0)
                recent_std = np.std(X[-20:], axis=0) if len(X) >= 20 else np.std(X, axis=0)
                
                for i, (val, avg, std, name) in enumerate(zip(current_data_point, recent_avg, recent_std, feature_names)):
                    if std > 0 and abs(val - avg) > 2 * std:  # More than 2 std deviations
                        reasons.append(name)
            
            return status, score, confidence, reasons
        except Exception as e:
            return "Error", 0.0, 0.0, []


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("AI System Monitor & Anomaly Detector - Enhanced with Paging Analysis")
        self.geometry("1200x750")
        
        # Data handling
        self.monitor_thread = None
        self.anomaly_detector = AnomalyDetector()
        
        # Historical data for plots
        self.history_cpu = collections.deque(maxlen=HISTORY_SIZE)
        self.history_mem = collections.deque(maxlen=HISTORY_SIZE)
        self.history_swap = collections.deque(maxlen=HISTORY_SIZE)
        self.history_page_faults = collections.deque(maxlen=HISTORY_SIZE)
        self.history_timestamps = collections.deque(maxlen=HISTORY_SIZE)
        self.anomalies_indices = []

        self._setup_ui()
        
        # Start monitoring
        self.monitor_thread = MonitorThread(self._on_metrics_update, interval=1.0)
        self.monitor_thread.start()

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_ui(self):
        # Grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar (Controls & Info) ---
        self.sidebar_frame = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(8, weight=1)

        # Title
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Resource Monitor", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Metrics Labels
        self.lbl_cpu = self._create_metric_label(1, "CPU Usage")
        self.lbl_mem = self._create_metric_label(2, "Memory Usage")
        self.lbl_disk = self._create_metric_label(3, "Disk Usage")
        self.lbl_net = self._create_metric_label(4, "Network I/O")
        
        # PAGING METRICS
        self.lbl_swap = self._create_metric_label(5, "Swap Usage")
        self.lbl_page_faults = self._create_metric_label(6, "Page Faults/s")

        # ML Status
        self.ml_frame = ctk.CTkFrame(self.sidebar_frame)
        self.ml_frame.grid(row=7, column=0, padx=10, pady=20, sticky="ew")
        ctk.CTkLabel(self.ml_frame, text="ML Analysis", font=ctk.CTkFont(weight="bold")).pack(pady=5)
        self.lbl_ml_status = ctk.CTkLabel(self.ml_frame, text="Status: Initializing...", text_color="gray")
        self.lbl_ml_status.pack(pady=2)
        self.lbl_ml_score = ctk.CTkLabel(self.ml_frame, text="Score: --")
        self.lbl_ml_score.pack(pady=2)
        self.lbl_ml_reasons = ctk.CTkLabel(self.ml_frame, text="", text_color="orange", wraplength=240)
        self.lbl_ml_reasons.pack(pady=2)

        self.btn_export = ctk.CTkButton(self.sidebar_frame, text="Export Data (CSV)", command=self._export_data, fg_color="green")
        self.btn_export.grid(row=8, column=0, padx=20, pady=(20, 20), sticky="n")

        # --- Main Content Area ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_rowconfigure(0, weight=2)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Charts Area - Now with 2 charts
        self.chart_frame = ctk.CTkFrame(self.main_frame)
        self.chart_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 20))
        self.chart_frame.grid_rowconfigure(0, weight=1)
        self.chart_frame.grid_columnconfigure(0, weight=1)
        self.chart_frame.grid_columnconfigure(1, weight=1)
        
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
        # Create TWO figures side by side
        
        # Chart 1: CPU & Memory
        chart1_frame = ctk.CTkFrame(self.chart_frame)
        chart1_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        self.fig1 = Figure(figsize=(4, 3.5), dpi=100)
        self.fig1.patch.set_facecolor('#2b2b2b')
        
        self.ax1 = self.fig1.add_subplot(111)
        self.ax1.set_facecolor('#2b2b2b')
        self.ax1.tick_params(axis='x', colors='white')
        self.ax1.tick_params(axis='y', colors='white')
        self.ax1.set_title("CPU & Memory", color='white')
        self.ax1.grid(True, color='#444444')
        
        self.line_cpu, = self.ax1.plot([], [], label='CPU %', color="#1eff00", linewidth=1.5)
        self.line_mem, = self.ax1.plot([], [], label='Memory %', color="#0400ff", linewidth=1.5)
        
        self.ax1.legend(facecolor='#2b2b2b', labelcolor='white', fontsize=8)
        
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=chart1_frame)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().pack(fill="both", expand=True)
        
        # Chart 2: Swap & Page Faults
        chart2_frame = ctk.CTkFrame(self.chart_frame)
        chart2_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        
        self.fig2 = Figure(figsize=(4, 3.5), dpi=100)
        self.fig2.patch.set_facecolor('#2b2b2b')
        
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2_twin = self.ax2.twinx()  # Secondary Y-axis
        
        self.ax2.set_facecolor('#2b2b2b')
        self.ax2.tick_params(axis='x', colors='white')
        self.ax2.tick_params(axis='y', colors='white')
        self.ax2_twin.tick_params(axis='y', colors='white')
        self.ax2.set_title("Memory Paging Metrics", color='white')
        self.ax2.grid(True, color='#444444')
        
        self.line_swap, = self.ax2.plot([], [], label='Swap %', color="#ff9500", linewidth=1.5)
        self.line_pf, = self.ax2_twin.plot([], [], label='Page Faults/s', color="#ff00ff", linewidth=1.5)
        
        self.ax2.set_ylabel('Swap %', color='white')
        self.ax2_twin.set_ylabel('Page Faults/s', color='white')
        
        # Combined legend
        lines = [self.line_swap, self.line_pf]
        labels = [l.get_label() for l in lines]
        self.ax2.legend(lines, labels, facecolor='#2b2b2b', labelcolor='white', fontsize=8, loc='upper left')
        
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=chart2_frame)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().pack(fill="both", expand=True)

    def _on_metrics_update(self, data):
        self.after(0, self._update_gui, data)

    def _update_gui(self, data):
        # Update Labels
        self.lbl_cpu.configure(text=f"{data.cpu_percent:.1f}%")
        self.lbl_mem.configure(text=f"{data.memory_percent:.1f}%")
        self.lbl_disk.configure(text=f"{data.disk_percent:.1f}%")
        self.lbl_net.configure(text=f"↑{data.network_sent:.1f} ↓{data.network_recv:.1f} MB/s")
        
        # PAGING METRICS
        self.lbl_swap.configure(text=f"{data.swap_percent:.1f}%")
        pf_display = f"{data.page_faults_per_sec:.0f}/s"
        if data.page_faults_per_sec > 1000:
            pf_display = f"{data.page_faults_per_sec/1000:.1f}k/s"
        self.lbl_page_faults.configure(text=pf_display)
        
        # Color coding for swap
        if data.swap_percent > 50:
            self.lbl_swap.configure(text_color="#f13636")  # Red
        elif data.swap_percent > 20:
            self.lbl_swap.configure(text_color="#ff9500")  # Orange
        else:
            self.lbl_swap.configure(text_color="cyan")

        # ML Detection with paging metrics
        self.anomaly_detector.add_data(data.to_array())
        status, score, conf, reasons = self.anomaly_detector.train_and_predict(data.to_array())
        
        self.lbl_ml_status.configure(text=status)
        self.lbl_ml_score.configure(text=f"Score: {score:.2f}")
        
        if reasons:
            reason_text = "Triggers: " + ", ".join(reasons)
            self.lbl_ml_reasons.configure(text=reason_text)
        else:
            self.lbl_ml_reasons.configure(text="")
        
        if status == "ANOMALY":
            self.lbl_ml_status.configure(text_color="#f13636")
            reason_detail = f" ({', '.join(reasons)})" if reasons else ""
            self._log_alert(f"ANOMALY: Score={score:.2f}{reason_detail}")
            
            # Special alert for memory pressure
            if "Swap" in reasons or "Page Faults" in reasons:
                self._log_alert("⚠ MEMORY PRESSURE detected - high paging activity")
        elif status == "Normal":
            self.lbl_ml_status.configure(text_color="#41f158")
        else:
            self.lbl_ml_status.configure(text_color="gray")

        # Update History & Charts
        self.history_cpu.append(data.cpu_percent)
        self.history_mem.append(data.memory_percent)
        self.history_swap.append(data.swap_percent)
        self.history_page_faults.append(data.page_faults_per_sec)
        self.history_timestamps.append(data.timestamp)
        
        x_data = range(len(self.history_cpu))
        
        # Chart 1: CPU & Memory
        self.line_cpu.set_data(x_data, self.history_cpu)
        self.line_mem.set_data(x_data, self.history_mem)
        self.ax1.set_xlim(0, max(HISTORY_SIZE, len(self.history_cpu)))
        self.ax1.set_ylim(0, 100)
        self.canvas1.draw()
        
        # Chart 2: Swap & Page Faults
        self.line_swap.set_data(x_data, self.history_swap)
        self.line_pf.set_data(x_data, self.history_page_faults)
        self.ax2.set_xlim(0, max(HISTORY_SIZE, len(self.history_swap)))
        self.ax2.set_ylim(0, 100)
        
        # Auto-scale page faults axis
        if len(self.history_page_faults) > 0:
            max_pf = max(self.history_page_faults) if max(self.history_page_faults) > 0 else 100
            self.ax2_twin.set_ylim(0, max_pf * 1.1)
        
        self.canvas2.draw()

        # Top Processes
        self._update_top_processes()

    def _update_top_processes(self):
        try:
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
                writer.writerow(["Timestamp", "CPU %", "Memory %", "Swap %", "Page Faults/s"])
                
                length = min(len(self.history_timestamps), len(self.history_cpu), 
                           len(self.history_mem), len(self.history_swap), len(self.history_page_faults))
                
                for i in range(length):
                    writer.writerow([
                        self.history_timestamps[i].strftime("%Y-%m-%d %H:%M:%S"),
                        self.history_cpu[i],
                        self.history_mem[i],
                        self.history_swap[i],
                        self.history_page_faults[i]
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