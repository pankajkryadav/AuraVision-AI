import tkinter as tk
import matplotlib
# Use TkAgg backend for embedding in Tkinter
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from datetime import datetime

class AuraChartDrawer:
    @staticmethod
    def apply_dark_theme(fig, ax):
        """
        Applies a cohesive neon-dark theme to Matplotlib charts.
        """
        fig.patch.set_facecolor("#1a1a24")  # Match CustomTkinter dark background
        ax.set_facecolor("#1a1a24")
        
        # Color ticks and labels
        ax.tick_params(colors="#a1a1b4", labelsize=8)
        ax.xaxis.label.set_color("#a1a1b4")
        ax.yaxis.label.set_color("#a1a1b4")
        ax.title.set_color("#ffffff")
        
        # Hide standard spines
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_color("#2e2e3f")
            
        # Grid settings
        ax.grid(True, color="#2e2e3f", linestyle="--", linewidth=0.5)

class FocusTrendChart:
    def __init__(self, parent_frame):
        self.parent = parent_frame
        
        # Create figure and axis
        self.fig = Figure(figsize=(5, 2.8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        AuraChartDrawer.apply_dark_theme(self.fig, self.ax)
        
        # Embed in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.configure(background="#1a1a24")
        
    def pack(self, **kwargs):
        self.canvas_widget.pack(**kwargs)
        
    def grid(self, **kwargs):
        self.canvas_widget.grid(**kwargs)
        
    def update_chart(self, mood_logs):
        """
        Expects a list of logs from MongoDB (sorted chronologically).
        """
        self.ax.clear()
        AuraChartDrawer.apply_dark_theme(self.fig, self.ax)
        self.ax.set_title("Focus & Attention Trend", fontsize=10, fontweight="bold", pad=10)
        self.ax.set_ylabel("Focus Score (%)", fontsize=8)
        self.ax.set_ylim(0, 105)
        
        if not mood_logs:
            self.ax.text(0.5, 0.5, "No Logs Available\nStart the monitor to collect data.", 
                         color="#6e6e80", ha="center", va="center", transform=self.ax.transAxes)
            self.canvas.draw()
            return
            
        times = []
        scores = []
        
        for idx, log in enumerate(mood_logs):
            t = log.get("timestamp")
            # Format time as HH:MM:SS
            if isinstance(t, datetime):
                times.append(t.strftime("%H:%M:%S"))
            else:
                times.append(str(idx))
            scores.append(log.get("focus_score", 50.0))
            
        # Select subset of labels to avoid overlapping x-ticks
        step = max(1, len(times) // 5)
        self.ax.set_xticks(range(0, len(times), step))
        self.ax.set_xticklabels([times[i] for i in range(0, len(times), step)], rotation=15)
        
        # Plot with cyan glowing line and marker
        self.ax.plot(scores, color="#00f3ff", linewidth=2.5, marker="o", markersize=4, 
                     markerfacecolor="#ffffff", markeredgecolor="#00f3ff", label="Focus Score")
                     
        # Fill area under line for a sleek modern look
        self.ax.fill_between(range(len(scores)), scores, color="#00f3ff", alpha=0.1)
        
        self.fig.tight_layout()
        self.canvas.draw()

class EmotionDistChart:
    def __init__(self, parent_frame):
        self.parent = parent_frame
        
        self.fig = Figure(figsize=(5, 2.8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        AuraChartDrawer.apply_dark_theme(self.fig, self.ax)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.configure(background="#1a1a24")
        
    def pack(self, **kwargs):
        self.canvas_widget.pack(**kwargs)
        
    def grid(self, **kwargs):
        self.canvas_widget.grid(**kwargs)
        
    def update_chart(self, emotion_counts):
        """
        Expects a dictionary of aggregate emotions: {'Focused': 23, 'Happy': 5, ...}
        """
        self.ax.clear()
        AuraChartDrawer.apply_dark_theme(self.fig, self.ax)
        self.ax.set_title("Emotion Distribution", fontsize=10, fontweight="bold", pad=10)
        
        if not emotion_counts or sum(emotion_counts.values()) == 0:
            self.ax.text(0.5, 0.5, "No Logs Available\nStart the monitor to collect data.", 
                         color="#6e6e80", ha="center", va="center", transform=self.ax.transAxes)
            self.canvas.draw()
            return
            
        labels = list(emotion_counts.keys())
        sizes = list(emotion_counts.values())
        
        # Color mapping for theme consistency
        color_map = {
            "Focused": "#00f3ff", # Cyan
            "Happy": "#00ff66",   # Neon Green
            "Stressed": "#ff0066",# Pink-Red
            "Tired": "#ffaa00"    # Orange
        }
        colors = [color_map.get(label, "#a1a1b4") for label in labels]
        
        # Plot styled donut chart
        wedges, texts, autotexts = self.ax.pie(
            sizes, 
            labels=labels, 
            autopct="%1.0f%%", 
            startangle=90, 
            colors=colors,
            textprops=dict(color="#a1a1b4", size=8),
            wedgeprops=dict(width=0.4, edgecolor="#1a1a24", linewidth=2)  # width < 1 creates donut holes
        )
        
        # Brighten percentage text inside slice
        for autotext in autotexts:
            autotext.set_color("#ffffff")
            autotext.set_weight("bold")
            autotext.set_size(8)
            
        self.fig.tight_layout()
        self.canvas.draw()
