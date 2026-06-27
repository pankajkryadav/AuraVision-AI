import customtkinter as ctk
import tkinter as tk
from PIL import Image
import threading
import time
from datetime import datetime

# Import local modules
from database import DatabaseManager
from vision_engine import VisionEngine
from nlp_engine import NLPSentimentEngine
from ai_companion import AICompanion
from charts import FocusTrendChart, EmotionDistChart
import pdf_app

# Configure CustomTkinter
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

import re

def parse_markdown(text):
    """
    Parses a markdown string to extract plain text and code blocks.
    Returns a list of tuples: [("text", text_content), ("code", language, code_content)]
    """
    if not text:
        return []
    parts = []
    current_pos = 0
    # Matches ```lang\ncode``` or ```code```
    pattern = r"```(\w*)\n(.*?)```"
    for match in re.finditer(pattern, text, re.DOTALL):
        start, end = match.span()
        if start > current_pos:
            parts.append(("text", text[current_pos:start]))
        lang = match.group(1).strip()
        code = match.group(2)
        parts.append(("code", lang, code))
        current_pos = end
    if current_pos < len(text):
        parts.append(("text", text[current_pos:]))
    return parts

def highlight_code(textbox, code_text):
    """
    Applies custom syntax highlighting to a ctk.CTkTextbox widget's raw text.
    """
    textbox.configure(state="normal")
    textbox.delete("1.0", "end")
    stripped_code = code_text.strip()
    textbox.insert("1.0", stripped_code)
    
    # Configure formatting tags on the underlying tkinter Text widget
    raw_text = textbox.textbox
    raw_text.tag_config("keyword", foreground="#ff0066", font=ctk.CTkFont(family="Consolas", size=11, weight="bold"))
    raw_text.tag_config("string", foreground="#00ff66", font=ctk.CTkFont(family="Consolas", size=11))
    raw_text.tag_config("comment", foreground="#6e6e80", font=ctk.CTkFont(family="Consolas", size=11, slant="italic"))
    raw_text.tag_config("number", foreground="#ffaa00", font=ctk.CTkFont(family="Consolas", size=11))
    raw_text.tag_config("builtin", foreground="#00f3ff", font=ctk.CTkFont(family="Consolas", size=11))
    
    # Regex rules
    keywords = r"\b(def|class|import|from|return|if|elif|else|for|while|try|except|finally|with|as|lambda|pass|break|continue|in|is|and|or|not|yield|global|nonlocal|const|let|var|function|async|await)\b"
    builtins = r"\b(print|self|None|True|False|int|float|str|list|dict|set|tuple|len|range|enumerate|zip|map|filter|sum|min|max|abs|round|open|__init__)\b"
    strings = r"(\".*?\"|'.*?')"
    numbers = r"\b(\d+)\b"
    comments = r"(#.*?$|//.*?$)"
    
    # Search and tag keywords
    for match in re.finditer(keywords, stripped_code):
        start_idx = f"1.0 + {match.start()} chars"
        end_idx = f"1.0 + {match.end()} chars"
        raw_text.tag_add("keyword", start_idx, end_idx)
        
    # Search and tag builtins
    for match in re.finditer(builtins, stripped_code):
        start_idx = f"1.0 + {match.start()} chars"
        end_idx = f"1.0 + {match.end()} chars"
        raw_text.tag_add("builtin", start_idx, end_idx)
        
    # Search and tag numbers
    for match in re.finditer(numbers, stripped_code):
        start_idx = f"1.0 + {match.start()} chars"
        end_idx = f"1.0 + {match.end()} chars"
        raw_text.tag_add("number", start_idx, end_idx)
        
    # Search and tag strings (applied after to override builtins/keywords inside strings)
    for match in re.finditer(strings, stripped_code):
        start_idx = f"1.0 + {match.start()} chars"
        end_idx = f"1.0 + {match.end()} chars"
        raw_text.tag_add("string", start_idx, end_idx)
        
    # Search and tag comments (applied last to override everything else)
    for match in re.finditer(comments, stripped_code, re.MULTILINE):
        start_idx = f"1.0 + {match.start()} chars"
        end_idx = f"1.0 + {match.end()} chars"
        raw_text.tag_add("comment", start_idx, end_idx)
        
    textbox.configure(state="disabled")

class AuraVisionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Configure Window
        self.title("AuraVision AI — Cognitive Health & Workspace Analytics Suite")
        self.geometry("1100x680")
        self.resizable(False, False)
        
        # Connection status cache
        self.db_connected = False
        self.ollama_connected = False
        self.user_id = None

        # Initialize Backend Core Systems
        self.db = DatabaseManager()
        
        self.nlp = NLPSentimentEngine()
        self.ai = AICompanion()
        
        # Note: VisionEngine will attempt to retrain its classifier using DB records inside its init
        self.vision = VisionEngine(db_manager=self.db, user_id=self.user_id)
        self.vision_running = False

        # Start background thread for connection checks
        self.connection_thread = threading.Thread(target=self._connection_check_loop, daemon=True)
        self.connection_thread.start()
        
        # UI State Variables
        self.active_tab = "dashboard"
        
        # Build UI layout
        self.create_layout()
        
        # Show default tab
        self.select_tab("dashboard")
        
        # Start periodic GUI updates
        self.update_gui_loop()
        
    def create_layout(self):
        # Configure Grid layout (1 row, 2 columns: Sidebar & Main Area)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1) # Sidebar (width is locked implicitly)
        self.grid_columnconfigure(1, weight=5) # Content Area
        
        # ==========================================
        # 1. SIDEBAR FRAME
        # ==========================================
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0, fg_color="#0f0f15")
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(8, weight=1) # Spacer row
        
        # Logo / Title
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="AURAVISION AI", 
                                       font=ctk.CTkFont(family="Helvetica", size=18, weight="bold"),
                                       text_color="#00f3ff")
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 5))
        
        self.sub_logo_label = ctk.CTkLabel(self.sidebar_frame, text="Cognitive Monitor & Companion", 
                                           font=ctk.CTkFont(family="Helvetica", size=9, slant="italic"),
                                           text_color="#6e6e80")
        self.sub_logo_label.grid(row=1, column=0, padx=20, pady=(0, 30))
        
        # Sidebar Navigation Buttons
        self.btn_dash = ctk.CTkButton(self.sidebar_frame, text="Dashboard", anchor="w",
                                      fg_color="transparent", text_color="#a1a1b4",
                                      command=lambda: self.select_tab("dashboard"))
        self.btn_dash.grid(row=2, column=0, padx=15, pady=8, sticky="ew")
        
        self.btn_diary = ctk.CTkButton(self.sidebar_frame, text="Cognitive Diary", anchor="w",
                                       fg_color="transparent", text_color="#a1a1b4",
                                       command=lambda: self.select_tab("diary"))
        self.btn_diary.grid(row=3, column=0, padx=15, pady=8, sticky="ew")
        
        self.btn_chat = ctk.CTkButton(self.sidebar_frame, text="Aura AI Companion", anchor="w",
                                      fg_color="transparent", text_color="#a1a1b4",
                                      command=lambda: self.select_tab("chat"))
        self.btn_chat.grid(row=4, column=0, padx=15, pady=8, sticky="ew")
        
        self.btn_calib = ctk.CTkButton(self.sidebar_frame, text="ML Calibration", anchor="w",
                                       fg_color="transparent", text_color="#a1a1b4",
                                       command=lambda: self.select_tab("calibration"))
        self.btn_calib.grid(row=5, column=0, padx=15, pady=8, sticky="ew")
        
        self.btn_charts = ctk.CTkButton(self.sidebar_frame, text="Analytics Hub", anchor="w",
                                        fg_color="transparent", text_color="#a1a1b4",
                                        command=lambda: self.select_tab("analytics"))
        self.btn_charts.grid(row=6, column=0, padx=15, pady=8, sticky="ew")
        
        self.btn_pdf = ctk.CTkButton(self.sidebar_frame, text="PDF Intelligence", anchor="w",
                                        fg_color="transparent", text_color="#a1a1b4",
                                        command=lambda: self.select_tab("pdf"))
        self.btn_pdf.grid(row=7, column=0, padx=15, pady=8, sticky="ew")
        
        # System Connection Indicators (in sidebar)
        self.status_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.status_frame.grid(row=9, column=0, padx=20, pady=20, sticky="s")
        
        self.lbl_db_status = ctk.CTkLabel(self.status_frame, text="MongoDB: Checking...", font=ctk.CTkFont(size=10), text_color="#ffaa00")
        self.lbl_db_status.pack(anchor="w", pady=2)
        
        self.lbl_ollama_status = ctk.CTkLabel(self.status_frame, text="Ollama LLM: Checking...", font=ctk.CTkFont(size=10), text_color="#ffaa00")
        self.lbl_ollama_status.pack(anchor="w", pady=2)
        
        self.lbl_ml_status = ctk.CTkLabel(self.status_frame, text="ML Mode: Rule-Based", font=ctk.CTkFont(size=10), text_color="#a1a1b4")
        self.lbl_ml_status.pack(anchor="w", pady=2)
        
        # ==========================================
        # 2. MAIN CONTENT FRAME container
        # ==========================================
        self.content_frame = ctk.CTkFrame(self, fg_color="#14141c", corner_radius=0)
        self.content_frame.grid(row=0, column=1, sticky="nsew")
        self.content_frame.grid_rowconfigure(0, weight=1)
        self.content_frame.grid_columnconfigure(0, weight=1)

        # Tab Frames Initialization
        self.frame_dashboard = self.init_dashboard_tab()
        self.frame_diary = self.init_diary_tab()
        self.frame_chat = self.init_chat_tab()
        self.frame_calibration = self.init_calibration_tab()
        self.frame_analytics = self.init_analytics_tab()
        self.frame_pdf = self.init_pdf_tab()

    def select_tab(self, tab_name):
        self.active_tab = tab_name
        
        # Hide all frames
        self.frame_dashboard.grid_remove()
        self.frame_diary.grid_remove()
        self.frame_chat.grid_remove()
        self.frame_calibration.grid_remove()
        self.frame_analytics.grid_remove()
        self.frame_pdf.grid_remove()
        
        # Reset sidebar buttons colors
        self.btn_dash.configure(fg_color="transparent", text_color="#a1a1b4")
        self.btn_diary.configure(fg_color="transparent", text_color="#a1a1b4")
        self.btn_chat.configure(fg_color="transparent", text_color="#a1a1b4")
        self.btn_calib.configure(fg_color="transparent", text_color="#a1a1b4")
        self.btn_charts.configure(fg_color="transparent", text_color="#a1a1b4")
        self.btn_pdf.configure(fg_color="transparent", text_color="#a1a1b4")
        
        # Show active frame & highlight button
        if tab_name == "dashboard":
            self.frame_dashboard.grid(row=0, column=0, sticky="nsew")
            self.btn_dash.configure(fg_color="#00f3ff", text_color="#0f0f15")
        elif tab_name == "diary":
            self.frame_diary.grid(row=0, column=0, sticky="nsew")
            self.btn_diary.configure(fg_color="#00f3ff", text_color="#0f0f15")
            self.load_diary_history()
        elif tab_name == "chat":
            self.frame_chat.grid(row=0, column=0, sticky="nsew")
            self.btn_chat.configure(fg_color="#00f3ff", text_color="#0f0f15")
            self.load_chat_history()
        elif tab_name == "calibration":
            self.frame_calibration.grid(row=0, column=0, sticky="nsew")
            self.btn_calib.configure(fg_color="#00f3ff", text_color="#0f0f15")
            self.update_calibration_sample_counts()
        elif tab_name == "analytics":
            self.frame_analytics.grid(row=0, column=0, sticky="nsew")
            self.btn_charts.configure(fg_color="#00f3ff", text_color="#0f0f15")
            self.refresh_analytics_plots()
        elif tab_name == "pdf":
            self.frame_pdf.grid(row=0, column=0, sticky="nsew")
            self.btn_pdf.configure(fg_color="#00f3ff", text_color="#0f0f15")

    # ==========================================
    # TAB 1: DASHBOARD
    # ==========================================
    def init_dashboard_tab(self):
        frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=3) # Camera / Left
        frame.grid_columnconfigure(1, weight=2) # Stats / Right
        frame.grid_rowconfigure(0, weight=1)
        
        # --- LEFT PANEL: CAMERA VIEW ---
        left_panel = ctk.CTkFrame(frame, fg_color="#1a1a24", corner_radius=15, border_width=1, border_color="#2e2e3f")
        left_panel.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        lbl_cam_title = ctk.CTkLabel(left_panel, text="Real-Time Workspace AI Feeds", 
                                     font=ctk.CTkFont(size=15, weight="bold"), text_color="#ffffff")
        lbl_cam_title.pack(anchor="w", padx=20, pady=(20, 10))
        
        # Display Box for Camera feed
        self.video_container = ctk.CTkFrame(left_panel, fg_color="#0f0f15", width=400, height=300)
        self.video_container.pack(padx=20, pady=10, fill="both", expand=True)
        self.video_container.pack_propagate(False)
        
        # Placeholder or Video Label
        self.video_label = ctk.CTkLabel(self.video_container, text="Workspace Monitor Standby\n\nClick 'Start AI Feed' below to run camera monitoring.",
                                        text_color="#6e6e80", font=ctk.CTkFont(size=12))
        self.video_label.pack(fill="both", expand=True)
        
        # Camera Actions
        self.btn_toggle_vision = ctk.CTkButton(left_panel, text="Start AI Feed", fg_color="#00ff66", text_color="#0f0f15", 
                                               font=ctk.CTkFont(weight="bold"), hover_color="#00dd55",
                                               command=self.toggle_vision_feed)
        self.btn_toggle_vision.pack(padx=20, pady=(10, 20), fill="x")
        
        # --- RIGHT PANEL: METRICS & CONTROLS ---
        right_panel = ctk.CTkFrame(frame, fg_color="#1a1a24", corner_radius=15, border_width=1, border_color="#2e2e3f")
        right_panel.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")
        
        lbl_metrics_title = ctk.CTkLabel(right_panel, text="Cognitive State HUD", 
                                         font=ctk.CTkFont(size=15, weight="bold"), text_color="#ffffff")
        lbl_metrics_title.pack(anchor="w", padx=20, pady=(20, 15))
        
        # Emotion Stat Display Card
        emotion_card = ctk.CTkFrame(right_panel, fg_color="#242435", corner_radius=10)
        emotion_card.pack(padx=20, pady=10, fill="x")
        
        lbl_emotion_text = ctk.CTkLabel(emotion_card, text="Detected State", font=ctk.CTkFont(size=10), text_color="#a1a1b4")
        lbl_emotion_text.pack(anchor="w", padx=15, pady=(8, 0))
        
        self.lbl_emotion_value = ctk.CTkLabel(emotion_card, text="Monitoring Standby", font=ctk.CTkFont(size=20, weight="bold"), text_color="#00f3ff")
        self.lbl_emotion_value.pack(anchor="w", padx=15, pady=(0, 8))
        
        # Focus score Progressbar Display
        focus_card = ctk.CTkFrame(right_panel, fg_color="#242435", corner_radius=10)
        focus_card.pack(padx=20, pady=10, fill="x")
        
        self.lbl_focus_text = ctk.CTkLabel(focus_card, text="Focus & Attention Index: 100%", font=ctk.CTkFont(size=11, weight="bold"), text_color="#ffffff")
        self.lbl_focus_text.pack(anchor="w", padx=15, pady=(8, 5))
        
        self.focus_bar = ctk.CTkProgressBar(focus_card, height=10, progress_color="#00f3ff", fg_color="#0f0f15")
        self.focus_bar.set(1.0)
        self.focus_bar.pack(fill="x", padx=15, pady=(0, 12))
        
        # Fatigue alert status card
        self.fatigue_card = ctk.CTkFrame(right_panel, fg_color="#242435", corner_radius=10)
        self.fatigue_card.pack(padx=20, pady=10, fill="x")
        
        self.lbl_fatigue_alert = ctk.CTkLabel(self.fatigue_card, text="Fatigue Watchdog: Normal", font=ctk.CTkFont(size=12, weight="bold"), text_color="#00ff66")
        self.lbl_fatigue_alert.pack(padx=15, pady=12, anchor="w")
        
        # AI Real-time Insights Card (Highlighted glassmorphism layout)
        self.insights_card = ctk.CTkFrame(right_panel, fg_color="#1e1b29", corner_radius=10, border_width=1, border_color="#8b5cf6")
        self.insights_card.pack(padx=20, pady=10, fill="x")
        
        lbl_insights_header = ctk.CTkLabel(self.insights_card, text="AURA LIVE AI ADVICE", font=ctk.CTkFont(size=10, weight="bold"), text_color="#8b5cf6")
        lbl_insights_header.pack(anchor="w", padx=15, pady=(8, 2))
        
        self.lbl_ai_insights = ctk.CTkLabel(
            self.insights_card, 
            text="Aura is analyzing your feed. Real-time workspace recommendations will appear here shortly...", 
            font=ctk.CTkFont(size=11, slant="italic"), 
            text_color="#ffffff", 
            wraplength=200, 
            justify="left"
        )
        self.lbl_ai_insights.pack(anchor="w", padx=15, pady=(2, 10))
        
        # Logged stats info card
        stats_card = ctk.CTkFrame(right_panel, fg_color="#0f0f15", corner_radius=10)
        stats_card.pack(padx=20, pady=(15, 20), fill="both", expand=True)
        
        self.lbl_blink_hud = ctk.CTkLabel(stats_card, text="Session Blinks (Active / 10s): 0", font=ctk.CTkFont(size=11), text_color="#a1a1b4")
        self.lbl_blink_hud.pack(anchor="w", padx=15, pady=(15, 5))
        
        self.lbl_mode_hud = ctk.CTkLabel(stats_card, text="Model Classification Mode: Dynamic ML", font=ctk.CTkFont(size=11), text_color="#a1a1b4")
        self.lbl_mode_hud.pack(anchor="w", padx=15, pady=5)
        
        self.lbl_info_hud = ctk.CTkLabel(stats_card, text="Note: Focus metrics are automatically archived in MongoDB every 10 seconds for analytics.", 
                                         font=ctk.CTkFont(size=9, slant="italic"), text_color="#6e6e80", wraplength=200, justify="left")
        self.lbl_info_hud.pack(anchor="sw", padx=15, pady=(20, 15), side="bottom")
        
        return frame

    def toggle_vision_feed(self):
        if not self.vision_running:
            self.vision.start()
            self.vision_running = True
            self.btn_toggle_vision.configure(text="Stop AI Feed", fg_color="#ff0066", hover_color="#dd0055")
            self.lbl_ai_insights.configure(text="Analyzing facial cues and focus levels...")
            self.schedule_insights_loop()
        else:
            self.vision.stop()
            self.vision_running = False
            self.video_label.configure(image=None, text="Workspace Monitor Standby\n\nClick 'Start AI Feed' below to run camera monitoring.")
            self.btn_toggle_vision.configure(text="Start AI Feed", fg_color="#00ff66", hover_color="#00dd55")
            self.lbl_emotion_value.configure(text="Monitoring Standby", text_color="#00f3ff")
            self.lbl_focus_text.configure(text="Focus & Attention Index: --")
            self.focus_bar.set(0)
            self.lbl_fatigue_alert.configure(text="Fatigue Watchdog: Normal", text_color="#00ff66")
            self.fatigue_card.configure(fg_color="#242435")
            self.lbl_ai_insights.configure(text="Aura is analyzing your feed. Real-time workspace recommendations will appear here shortly...", font=ctk.CTkFont(size=11, slant="italic"))

    def schedule_insights_loop(self):
        if self.vision_running:
            self.update_ai_insights_task()
            self.after(12000, self.schedule_insights_loop)

    def update_ai_insights_task(self):
        if not self.vision_running:
            return
            
        metrics = self.vision.latest_metrics.copy()
        
        def run_insights():
            advice = self.ai.generate_visual_advice(metrics)
            self.after(10, lambda: self.lbl_ai_insights.configure(text=advice, font=ctk.CTkFont(size=11, weight="bold")))
            
        threading.Thread(target=run_insights, daemon=True).start()

    # ==========================================
    # TAB 2: COGNITIVE DIARY
    # ==========================================
    def init_diary_tab(self):
        frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=3) # Text box
        frame.grid_columnconfigure(1, weight=2) # Sentiment HUD & History
        frame.grid_rowconfigure(0, weight=1)
        
        # --- LEFT PANEL: DIARY CANVAS ---
        left_panel = ctk.CTkFrame(frame, fg_color="#1a1a24", corner_radius=15, border_width=1, border_color="#2e2e3f")
        left_panel.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        lbl_diary_title = ctk.CTkLabel(left_panel, text="Cognitive Health Diary", 
                                      font=ctk.CTkFont(size=15, weight="bold"), text_color="#ffffff")
        lbl_diary_title.pack(anchor="w", padx=20, pady=(20, 5))
        
        lbl_diary_sub = ctk.CTkLabel(left_panel, text="Write about your focus, stress points, or what you worked on today.", 
                                     font=ctk.CTkFont(size=10), text_color="#a1a1b4")
        lbl_diary_sub.pack(anchor="w", padx=20, pady=(0, 15))
        
        # Large textbox
        self.diary_textbox = ctk.CTkTextbox(left_panel, font=ctk.CTkFont(size=12), fg_color="#0f0f15", border_width=1, border_color="#2e2e3f")
        self.diary_textbox.pack(padx=20, pady=10, fill="both", expand=True)
        
        # Analyze Button
        self.btn_analyze_diary = ctk.CTkButton(left_panel, text="Log Diary & Run NLP Sentiment Model", fg_color="#00f3ff", text_color="#0f0f15",
                                               font=ctk.CTkFont(weight="bold"), hover_color="#00ccdd",
                                               command=self.analyze_and_save_diary)
        self.btn_analyze_diary.pack(padx=20, pady=20, fill="x")
        
        # --- RIGHT PANEL: SENTIMENT FEEDBACK & LOGS ---
        right_panel = ctk.CTkFrame(frame, fg_color="#1a1a24", corner_radius=15, border_width=1, border_color="#2e2e3f")
        right_panel.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")
        
        lbl_nlp_title = ctk.CTkLabel(right_panel, text="NLP Sentiment Feedback", 
                                     font=ctk.CTkFont(size=14, weight="bold"), text_color="#ffffff")
        lbl_nlp_title.pack(anchor="w", padx=20, pady=(20, 15))
        
        # Sentiment score display
        self.nlp_card = ctk.CTkFrame(right_panel, fg_color="#242435", corner_radius=10)
        self.nlp_card.pack(padx=20, pady=10, fill="x")
        
        self.lbl_nlp_polarity = ctk.CTkLabel(self.nlp_card, text="Polarity Score: --", font=ctk.CTkFont(size=11), text_color="#a1a1b4")
        self.lbl_nlp_polarity.pack(anchor="w", padx=15, pady=(10, 2))
        
        self.lbl_nlp_label = ctk.CTkLabel(self.nlp_card, text="Analyzed Tone: Standby", font=ctk.CTkFont(size=16, weight="bold"), text_color="#ffffff")
        self.lbl_nlp_label.pack(anchor="w", padx=15, pady=(2, 10))
        
        # History section
        lbl_hist_title = ctk.CTkLabel(right_panel, text="Recent Journal Logs", 
                                      font=ctk.CTkFont(size=12, weight="bold"), text_color="#ffffff")
        lbl_hist_title.pack(anchor="w", padx=20, pady=(15, 5))
        
        self.diary_history_box = ctk.CTkTextbox(right_panel, fg_color="#0f0f15", font=ctk.CTkFont(size=10), border_width=1, border_color="#2e2e3f")
        self.diary_history_box.pack(padx=20, pady=(5, 20), fill="both", expand=True)
        self.diary_history_box.configure(state="disabled")
        
        return frame

    def analyze_and_save_diary(self):
        text = self.diary_textbox.get("1.0", "end-1c").strip()
        if not text:
            return
            
        # Run local TF-IDF sentiment classifier
        polarity, label = self.nlp.analyze_document(text)
        
        # Format polarity string
        self.lbl_nlp_polarity.configure(text=f"Polarity Score: {polarity:.2f}")
        self.lbl_nlp_label.configure(text=f"Analyzed Tone: {label}")
        
        # Color styling based on sentiment
        if label == "Positive":
            self.nlp_card.configure(fg_color="#103520") # subtle forest green
            self.lbl_nlp_label.configure(text_color="#00ff66")
        elif label == "Negative":
            self.nlp_card.configure(fg_color="#3a1020") # subtle burgundy
            self.lbl_nlp_label.configure(text_color="#ff0066")
        else:
            self.nlp_card.configure(fg_color="#242435") # standard dark
            self.lbl_nlp_label.configure(text_color="#ffffff")
            
        # Log diary entry in MongoDB
        if self.db and self.user_id:
            self.db.save_journal_entry(self.user_id, text, polarity, label)
            
        # Clear typing area
        self.diary_textbox.delete("1.0", "end")
        
        # Refresh history list
        self.load_diary_history()
        
    def load_diary_history(self):
        if not self.db or not self.user_id:
            return
            
        logs = self.db.get_journal_history(self.user_id, limit=5)
        
        self.diary_history_box.configure(state="normal")
        self.diary_history_box.delete("1.0", "end")
        
        if not logs:
            self.diary_history_box.insert("end", "No journal logs stored in MongoDB yet.\n")
        else:
            for log in logs:
                t = log.get("timestamp")
                t_str = t.strftime("%b %d, %H:%M") if isinstance(t, datetime) else "Date Error"
                lbl = log.get("sentiment_label", "Neutral")
                txt = log.get("entry_text", "")
                
                # Trim entry text preview
                preview = txt if len(txt) < 80 else txt[:77] + "..."
                
                self.diary_history_box.insert("end", f"[{t_str}] Sentiment: {lbl}\n")
                self.diary_history_box.insert("end", f"\"{preview}\"\n")
                self.diary_history_box.insert("end", "-" * 35 + "\n")
                
        self.diary_history_box.configure(state="disabled")

    # ==========================================
    # TAB 3: AURA AI COMPANION (CHAT)
    # ==========================================
    def init_chat_tab(self):
        frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        
        main_panel = ctk.CTkFrame(frame, fg_color="#1a1a24", corner_radius=15, border_width=1, border_color="#2e2e3f")
        main_panel.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        main_panel.grid_rowconfigure(1, weight=1) # Chat history area
        main_panel.grid_columnconfigure(0, weight=1)
        
        # Top Title bar
        title_bar = ctk.CTkFrame(main_panel, fg_color="#0f0f15", height=50, corner_radius=10)
        title_bar.grid(row=0, column=0, padx=15, pady=(15, 5), sticky="ew")
        title_bar.pack_propagate(False)
        
        lbl_chat_title = ctk.CTkLabel(title_bar, text="Aura Workspace Counseling Desk", font=ctk.CTkFont(size=14, weight="bold"), text_color="#00f3ff")
        lbl_chat_title.pack(side="left", padx=15, pady=10)
        
        self.lbl_context_hud = ctk.CTkLabel(title_bar, text="Live Context feeding active...", font=ctk.CTkFont(size=10, slant="italic"), text_color="#a1a1b4")
        self.lbl_context_hud.pack(side="right", padx=15, pady=10)
        
        # Chat Messages Scrollable Frame (Modern Bubble Layout)
        self.chat_scroll = ctk.CTkScrollableFrame(main_panel, fg_color="#0f0f15", border_width=1, border_color="#2e2e3f")
        self.chat_scroll.grid(row=1, column=0, padx=15, pady=5, sticky="nsew")
        
        # Bottom controls for sending messages
        input_bar = ctk.CTkFrame(main_panel, fg_color="transparent")
        input_bar.grid(row=2, column=0, padx=15, pady=15, sticky="ew")
        input_bar.grid_columnconfigure(0, weight=1)
        
        self.chat_input = ctk.CTkEntry(input_bar, placeholder_text="Type a message or ask for a workspace feedback...",
                                      fg_color="#0f0f15", border_color="#2e2e3f", font=ctk.CTkFont(size=12))
        self.chat_input.grid(row=0, column=0, padx=(0, 10), sticky="ew")
        self.chat_input.bind("<Return>", lambda event: self.send_chat_message())
        
        btn_send = ctk.CTkButton(input_bar, text="Send Message", fg_color="#00f3ff", text_color="#0f0f15", width=120,
                                 font=ctk.CTkFont(weight="bold"), hover_color="#00ccdd",
                                 command=self.send_chat_message)
        btn_send.grid(row=0, column=1, sticky="e")
        
        return frame

    def send_chat_message(self):
        text = self.chat_input.get().strip()
        if not text:
            return
            
        self.chat_input.delete(0, "end")
        
        current_mood = self.vision.latest_metrics["emotion"]
        # Polarity check on current message
        _, current_sent = self.nlp.analyze_sentence(text)
        
        # Append User Message to DB
        if self.db and self.user_id:
            self.db.save_chat_message(self.user_id, "user", text, current_mood)
            
        # Refresh history UI immediately (so user message displays right away)
        self.load_chat_history()
        
        # Disable inputs while processing AI
        self.chat_input.configure(state="disabled")
        
        # Fetch history records to supply to Ollama
        history_logs = self.db.get_chat_history(self.user_id, limit=6) if self.db else []
        
        # Generate response in background thread to prevent GUI freezing
        def run_ai_task():
            reply = self.ai.generate_response(text, current_mood, current_sent, history_logs)
            
            # Save AI response to DB
            if self.db and self.user_id:
                self.db.save_chat_message(self.user_id, "assistant", reply, current_mood)
                
            # Update GUI in main thread
            self.after(10, self.complete_chat_response)
            
        threading.Thread(target=run_ai_task, daemon=True).start()

    def complete_chat_response(self):
        self.chat_input.configure(state="normal")
        self.chat_input.focus()
        self.load_chat_history()

    def load_chat_history(self):
        # Fallback if DB or user is not initialized
        if not self.db or not self.user_id:
            for widget in self.chat_scroll.winfo_children():
                widget.destroy()
            
            # Welcome message frame
            container = ctk.CTkFrame(self.chat_scroll, fg_color="transparent")
            container.pack(fill="x", pady=6, padx=10)
            
            header = ctk.CTkLabel(container, text="Aura", font=ctk.CTkFont(size=9, weight="bold"), text_color="#a1a1b4")
            header.pack(anchor="w", padx=10, pady=(0, 2))
            
            bubble = ctk.CTkFrame(container, fg_color="#242435", corner_radius=12)
            bubble.pack(anchor="w", padx=5)
            
            lbl = ctk.CTkLabel(
                bubble, 
                text="Hello! I'm Aura, your adaptive workspace counselor. I'm connected to your real-time face metrics and journal sentiment. Feel free to talk to me about your studies, stress levels, or asks for code layout tips!", 
                font=ctk.CTkFont(size=12), 
                text_color="#ffffff", 
                justify="left", 
                wraplength=550
            )
            lbl.pack(padx=12, pady=10)
            return
            
        chats = self.db.get_chat_history(self.user_id, limit=30)
        
        # Clear existing bubbles
        for widget in self.chat_scroll.winfo_children():
            widget.destroy()
            
        if not chats:
            # Welcome message from AI
            container = ctk.CTkFrame(self.chat_scroll, fg_color="transparent")
            container.pack(fill="x", pady=6, padx=10)
            
            header = ctk.CTkLabel(container, text="Aura", font=ctk.CTkFont(size=9, weight="bold"), text_color="#a1a1b4")
            header.pack(anchor="w", padx=10, pady=(0, 2))
            
            bubble = ctk.CTkFrame(container, fg_color="#242435", corner_radius=12)
            bubble.pack(anchor="w", padx=5)
            
            lbl = ctk.CTkLabel(
                bubble, 
                text="Hello! I'm Aura, your adaptive workspace counselor. I'm connected to your real-time face metrics and journal sentiment. Feel free to talk to me about your studies, stress levels, or asks for code layout tips!", 
                font=ctk.CTkFont(size=12), 
                text_color="#ffffff", 
                justify="left", 
                wraplength=550
            )
            lbl.pack(padx=12, pady=10)
        else:
            for c in chats:
                role = c.get("role", "assistant")
                is_user = (role == "user")
                msg = c.get("message", "")
                mood = c.get("user_mood_context", "Neutral")
                
                # Container for this message
                container = ctk.CTkFrame(self.chat_scroll, fg_color="transparent")
                container.pack(fill="x", pady=6, padx=10)
                
                # Header with name and metadata
                if is_user:
                    header_text = f"You (State: {mood})"
                else:
                    header_text = "Aura"
                    
                header = ctk.CTkLabel(container, text=header_text, font=ctk.CTkFont(size=9, weight="bold"), text_color="#a1a1b4")
                header.pack(anchor="e" if is_user else "w", padx=10, pady=(0, 2))
                
                # Message Bubble
                bubble_color = "#005f73" if is_user else "#242435"
                bubble = ctk.CTkFrame(container, fg_color=bubble_color, corner_radius=12)
                bubble.pack(anchor="e" if is_user else "w", padx=5)
                
                # Parse markdown content (for code blocks)
                parts = parse_markdown(msg)
                for part_type, *part_data in parts:
                    if part_type == "text":
                        text_content = part_data[0].strip()
                        if text_content:
                            lbl = ctk.CTkLabel(
                                bubble, 
                                text=text_content, 
                                font=ctk.CTkFont(size=12), 
                                text_color="#ffffff", 
                                justify="left", 
                                wraplength=550
                            )
                            lbl.pack(padx=12, pady=8, anchor="w")
                    elif part_type == "code":
                        lang = part_data[0]
                        code_content = part_data[1]
                        
                        # Create container for code block to look premium
                        code_container = ctk.CTkFrame(bubble, fg_color="#0b0b10", corner_radius=6, border_width=1, border_color="#2e2e3f")
                        code_container.pack(fill="x", padx=10, pady=6, anchor="w")
                        
                        # Optional language header
                        lang_header = ctk.CTkLabel(
                            code_container, 
                            text=f" {lang.upper() if lang else 'CODE'} ", 
                            font=ctk.CTkFont(family="Consolas", size=9, weight="bold"), 
                            text_color="#6e6e80"
                        )
                        lang_header.pack(anchor="w", padx=8, pady=(4, 0))
                        
                        # Calculate text box height based on number of lines
                        num_lines = len(code_content.strip().split('\n'))
                        box_height = min(250, max(45, num_lines * 18))
                        
                        # Text box for code rendering
                        code_box = ctk.CTkTextbox(
                            code_container, 
                            font=ctk.CTkFont(family="Consolas", size=11), 
                            fg_color="#0b0b10",
                            border_width=0,
                            height=box_height
                        )
                        code_box.pack(fill="x", padx=6, pady=4)
                        
                        # Apply syntax highlighting
                        highlight_code(code_box, code_content)
                        
        # Scroll to bottom
        self.chat_scroll.update_idletasks()
        self.chat_scroll._parent_canvas.yview_moveto(1.0)

    # ==========================================
    # TAB 4: ML CALIBRATION
    # ==========================================
    def init_calibration_tab(self):
        frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(0, weight=1)
        
        panel = ctk.CTkFrame(frame, fg_color="#1a1a24", corner_radius=15, border_width=1, border_color="#2e2e3f")
        panel.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        panel.grid_columnconfigure(0, weight=3) # Controls
        panel.grid_columnconfigure(1, weight=2) # Info & Clear
        panel.grid_rowconfigure(0, weight=1)
        
        # --- LEFT COL: CALIBRATION TRIGGER CONTROLS ---
        left_side = ctk.CTkFrame(panel, fg_color="transparent")
        left_side.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        lbl_calib_title = ctk.CTkLabel(left_side, text="Dynamic ML Calibration Panel", 
                                       font=ctk.CTkFont(size=15, weight="bold"), text_color="#00f3ff")
        lbl_calib_title.pack(anchor="w", pady=(10, 5))
        
        lbl_calib_desc = ctk.CTkLabel(left_side, text="Train your custom scikit-learn model in real-time.\nSelect an emotion state, click record, and hold that expression for 5 seconds.", 
                                      font=ctk.CTkFont(size=10), text_color="#a1a1b4", justify="left")
        lbl_calib_desc.pack(anchor="w", pady=(0, 20))
        
        # Selector
        lbl_sel_text = ctk.CTkLabel(left_side, text="Select Target Emotion to Record:", font=ctk.CTkFont(size=11, weight="bold"))
        lbl_sel_text.pack(anchor="w", pady=5)
        
        self.combo_emotions = ctk.CTkComboBox(left_side, values=["Focused", "Happy", "Stressed"], fg_color="#0f0f15", border_color="#2e2e3f")
        self.combo_emotions.pack(anchor="w", fill="x", pady=(0, 20))
        
        # Start button
        self.btn_run_calibration = ctk.CTkButton(left_side, text="Record Expression (5s)", fg_color="#ffaa00", text_color="#0f0f15",
                                                 font=ctk.CTkFont(weight="bold"), hover_color="#dd9900",
                                                 command=self.trigger_calibration)
        self.btn_run_calibration.pack(fill="x", pady=10)
        
        # Progress label
        self.lbl_calib_progress = ctk.CTkLabel(left_side, text="Status: Ready to record.", font=ctk.CTkFont(size=12, weight="bold"), text_color="#ffaa00")
        self.lbl_calib_progress.pack(anchor="w", pady=15)
        
        # --- RIGHT COL: STATS & MODEL RETRAIN ---
        right_side = ctk.CTkFrame(panel, fg_color="#0f0f15", corner_radius=10)
        right_side.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        lbl_stats_title = ctk.CTkLabel(right_side, text="MongoDB Calibration Dataset Size", font=ctk.CTkFont(size=12, weight="bold"), text_color="#ffffff")
        lbl_stats_title.pack(anchor="w", padx=15, pady=(15, 10))
        
        self.lbl_count_focus = ctk.CTkLabel(right_side, text="Focused samples: 0", font=ctk.CTkFont(size=11), text_color="#a1a1b4")
        self.lbl_count_focus.pack(anchor="w", padx=15, pady=3)
        
        self.lbl_count_happy = ctk.CTkLabel(right_side, text="Happy samples: 0", font=ctk.CTkFont(size=11), text_color="#a1a1b4")
        self.lbl_count_happy.pack(anchor="w", padx=15, pady=3)
        
        self.lbl_count_stress = ctk.CTkLabel(right_side, text="Stressed samples: 0", font=ctk.CTkFont(size=11), text_color="#a1a1b4")
        self.lbl_count_stress.pack(anchor="w", padx=15, pady=3)
        
        self.lbl_model_status_panel = ctk.CTkLabel(right_side, text="Classifier Status: Not Trained (Rule Fallback)", 
                                                  font=ctk.CTkFont(size=11, weight="bold"), text_color="#ff0066")
        self.lbl_model_status_panel.pack(anchor="w", padx=15, pady=(20, 10))
        
        btn_clear_calib = ctk.CTkButton(right_side, text="Wipe Calibration DB Records", fg_color="#3a1020", text_color="#ffffff",
                                        font=ctk.CTkFont(size=10), hover_color="#551525",
                                        command=self.clear_calibration_data)
        btn_clear_calib.pack(padx=15, pady=20, fill="x", side="bottom")
        
        return frame

    def update_calibration_sample_counts(self):
        if not self.db or not self.user_id:
            return
            
        data = self.db.get_all_calibration_data(self.user_id)
        
        counts = {"Focused": 0, "Happy": 0, "Stressed": 0}
        for record in data:
            emotion = record["emotion"]
            features = record["features"]
            if emotion in counts:
                counts[emotion] = len(features)
                
        self.lbl_count_focus.configure(text=f"Focused samples: {counts['Focused']}")
        self.lbl_count_happy.configure(text=f"Happy samples: {counts['Happy']}")
        self.lbl_count_stress.configure(text=f"Stressed samples: {counts['Stressed']}")
        
        # Check if vision engine is trained on ML
        if self.vision.is_ml_trained:
            self.lbl_model_status_panel.configure(text="Classifier Status: Trained (RandomForest active)", text_color="#00ff66")
            self.lbl_ml_status.configure(text="ML Mode: RandomForest", text_color="#00ff66")
        else:
            self.lbl_model_status_panel.configure(text="Classifier Status: Not Trained (Need 2+ classes with samples)", text_color="#ff0066")
            self.lbl_ml_status.configure(text="ML Mode: Rule-Based", text_color="#a1a1b4")

    def trigger_calibration(self):
        if not self.vision_running:
            self.lbl_calib_progress.configure(text="Error: Turn on AI Feed in Dashboard first!", text_color="#ff0066")
            return
            
        emotion = self.combo_emotions.get()
        self.btn_run_calibration.configure(state="disabled")
        self.combo_emotions.configure(state="disabled")
        
        # Start recording coordinates in vision engine
        self.vision.start_calibration(emotion)
        
        # Countdown loop using tkinter callbacks
        seconds_left = 5
        
        def countdown():
            nonlocal seconds_left
            if seconds_left > 0:
                self.lbl_calib_progress.configure(text=f"Recording facial features: {seconds_left}s remaining...", text_color="#ffaa00")
                seconds_left -= 1
                self.after(1000, countdown)
            else:
                # Finish recording
                success = self.vision.stop_and_save_calibration()
                
                self.btn_run_calibration.configure(state="normal")
                self.combo_emotions.configure(state="normal")
                
                if success:
                    self.lbl_calib_progress.configure(text="Calibration successful! scikit-learn model retrained.", text_color="#00ff66")
                else:
                    self.lbl_calib_progress.configure(text="Error saving features to MongoDB.", text_color="#ff0066")
                    
                self.update_calibration_sample_counts()
                
        countdown()

    def clear_calibration_data(self):
        if not self.db or not self.user_id:
            return
            
        self.db.clear_calibration_data(self.user_id)
        self.vision.retrain_classifier()
        self.update_calibration_sample_counts()
        self.lbl_calib_progress.configure(text="Status: Calibration database cleared.", text_color="#ffaa00")

    # ==========================================
    # TAB 5: ANALYTICS HUB
    # ==========================================
    def init_analytics_tab(self):
        frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(0, weight=1)
        
        panel = ctk.CTkFrame(frame, fg_color="#1a1a24", corner_radius=15, border_width=1, border_color="#2e2e3f")
        panel.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        panel.grid_rowconfigure(0, weight=1)
        panel.grid_rowconfigure(1, weight=0)
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_columnconfigure(1, weight=1)
        
        # Embedded Matplotlib Chart 1 (Focus Line Graph)
        chart_frame_left = ctk.CTkFrame(panel, fg_color="#1a1a24")
        chart_frame_left.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
        
        self.focus_trend_chart = FocusTrendChart(chart_frame_left)
        self.focus_trend_chart.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Embedded Matplotlib Chart 2 (Emotion Distribution Donut)
        chart_frame_right = ctk.CTkFrame(panel, fg_color="#1a1a24")
        chart_frame_right.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")
        
        self.emotion_dist_chart = EmotionDistChart(chart_frame_right)
        self.emotion_dist_chart.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Refresh Button bottom
        btn_refresh_charts = ctk.CTkButton(panel, text="Fetch Latest Logs & Reload Analytics", fg_color="#00f3ff", text_color="#0f0f15",
                                           font=ctk.CTkFont(weight="bold"), hover_color="#00ccdd",
                                           command=self.refresh_analytics_plots)
        btn_refresh_charts.grid(row=1, column=0, columnspan=2, pady=(0, 20), padx=20, sticky="ew")
        
        return frame

    def refresh_analytics_plots(self):
        if not self.db or not self.user_id:
            return
            
        # Get last 50 logs for focus line graph
        mood_logs = self.db.get_mood_history(self.user_id, limit=50)
        self.focus_trend_chart.update_chart(mood_logs)
        
        # Get emotion aggregate distribution
        emotion_counts = self.db.get_aggregate_emotions(self.user_id, hours=24)
        self.emotion_dist_chart.update_chart(emotion_counts)

    # ==========================================
    # TAB 6: PDF INTELLIGENCE
    # ==========================================
    def init_pdf_tab(self):
        frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=2)
        frame.grid_rowconfigure(0, weight=1)
        
        # Left Panel (File Upload and Output)
        left_panel = ctk.CTkFrame(frame, fg_color="#1a1a24", corner_radius=15, border_width=1, border_color="#2e2e3f")
        left_panel.grid(row=0, column=0, padx=(20, 10), pady=20, sticky="nsew")
        
        lbl_pdf_title = ctk.CTkLabel(left_panel, text="Local Massive PDF Intelligence", 
                                     font=ctk.CTkFont(size=15, weight="bold"), text_color="#ffffff")
        lbl_pdf_title.pack(anchor="w", padx=20, pady=(20, 10))
        
        self.lbl_pdf_file = ctk.CTkLabel(left_panel, text="No file selected", font=ctk.CTkFont(size=11), text_color="#a1a1b4", wraplength=250)
        self.lbl_pdf_file.pack(anchor="w", padx=20, pady=(0, 10))
        
        btn_select_pdf = ctk.CTkButton(left_panel, text="Select PDF File", command=self.select_pdf_file)
        btn_select_pdf.pack(padx=20, pady=5, fill="x")
        
        btn_process_pdf = ctk.CTkButton(left_panel, text="⚙️ Index & Analyze Document", fg_color="#00f3ff", text_color="#0f0f15", font=ctk.CTkFont(weight="bold"), command=self.process_pdf)
        btn_process_pdf.pack(padx=20, pady=5, fill="x")
        
        lbl_status_title = ctk.CTkLabel(left_panel, text="System Processing Log", font=ctk.CTkFont(size=12, weight="bold"))
        lbl_status_title.pack(anchor="w", padx=20, pady=(15, 5))
        
        self.txt_pdf_status = ctk.CTkTextbox(left_panel, height=100, fg_color="#0f0f15", font=ctk.CTkFont(size=11))
        self.txt_pdf_status.pack(padx=20, pady=5, fill="x")
        
        # Right Panel (Tabs for Notes, Mindmap, Chat)
        right_panel = ctk.CTkTabview(frame, fg_color="#1a1a24", segmented_button_selected_color="#00f3ff", segmented_button_selected_hover_color="#00ccdd")
        right_panel.grid(row=0, column=1, padx=(10, 20), pady=20, sticky="nsew")
        
        tab_notes = right_panel.add("📝 Structured Notes")
        tab_mindmap = right_panel.add("🗺️ Visual Mindmap")
        tab_chat = right_panel.add("💬 Ask Anything")
        
        # Notes Tab
        btn_gen_notes = ctk.CTkButton(tab_notes, text="✨ Generate Study Notes", command=self.gen_pdf_notes)
        btn_gen_notes.pack(padx=10, pady=10, fill="x")
        self.txt_pdf_notes = ctk.CTkTextbox(tab_notes, fg_color="#0f0f15", font=ctk.CTkFont(size=12))
        self.txt_pdf_notes.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Mindmap Tab
        btn_gen_mindmap = ctk.CTkButton(tab_mindmap, text="🌿 Render Hierarchical Mindmap", command=self.gen_pdf_mindmap)
        btn_gen_mindmap.pack(padx=10, pady=10, fill="x")
        self.txt_pdf_mindmap = ctk.CTkTextbox(tab_mindmap, fg_color="#0f0f15", font=ctk.CTkFont(size=12))
        self.txt_pdf_mindmap.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Chat Tab
        self.txt_pdf_chat_history = ctk.CTkTextbox(tab_chat, fg_color="#0f0f15", font=ctk.CTkFont(size=12))
        self.txt_pdf_chat_history.pack(padx=10, pady=10, fill="both", expand=True)
        self.txt_pdf_chat_history.configure(state="disabled")
        
        chat_input_frame = ctk.CTkFrame(tab_chat, fg_color="transparent")
        chat_input_frame.pack(padx=10, pady=5, fill="x")
        self.entry_pdf_chat = ctk.CTkEntry(chat_input_frame, placeholder_text="Ask a question about the PDF...", font=ctk.CTkFont(size=12))
        self.entry_pdf_chat.pack(side="left", fill="x", expand=True, padx=(0, 10))
        btn_pdf_chat_send = ctk.CTkButton(chat_input_frame, text="Send", width=80, fg_color="#00f3ff", text_color="#0f0f15", font=ctk.CTkFont(weight="bold"), command=self.send_pdf_chat)
        btn_pdf_chat_send.pack(side="right")
        self.entry_pdf_chat.bind("<Return>", lambda event: self.send_pdf_chat())
        
        return frame

    def select_pdf_file(self):
        filepath = ctk.filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if filepath:
            self.pdf_filepath = filepath
            self.lbl_pdf_file.configure(text=filepath)

    def process_pdf(self):
        if not hasattr(self, 'pdf_filepath') or not self.pdf_filepath:
            self.txt_pdf_status.delete("1.0", "end")
            self.txt_pdf_status.insert("end", "❌ Please select a valid PDF file first.\n")
            return
            
        self.txt_pdf_status.delete("1.0", "end")
        self.txt_pdf_status.insert("end", "Indexing and analyzing document...\n")
        
        def task():
            class DummyFile:
                def __init__(self, name):
                    self.name = name
            res = pdf_app.extract_and_chunk_pdf(DummyFile(self.pdf_filepath))
            self.after(10, lambda: [self.txt_pdf_status.delete("1.0", "end"), self.txt_pdf_status.insert("end", res + "\n")])
            
        threading.Thread(target=task, daemon=True).start()

    def gen_pdf_notes(self):
        self.txt_pdf_notes.delete("1.0", "end")
        self.txt_pdf_notes.insert("end", "Generating structured study notes... (This might take a while)\n")
        def task():
            res = pdf_app.generate_notes()
            self.after(10, lambda: [self.txt_pdf_notes.delete("1.0", "end"), self.txt_pdf_notes.insert("end", res)])
        threading.Thread(target=task, daemon=True).start()

    def gen_pdf_mindmap(self):
        self.txt_pdf_mindmap.delete("1.0", "end")
        self.txt_pdf_mindmap.insert("end", "Generating hierarchical mindmap... (This might take a while)\n")
        def task():
            res = pdf_app.generate_mindmap()
            self.after(10, lambda: [self.txt_pdf_mindmap.delete("1.0", "end"), self.txt_pdf_mindmap.insert("end", res)])
        threading.Thread(target=task, daemon=True).start()

    def send_pdf_chat(self):
        msg = self.entry_pdf_chat.get().strip()
        if not msg: return
        self.entry_pdf_chat.delete(0, "end")
        
        self.txt_pdf_chat_history.configure(state="normal")
        self.txt_pdf_chat_history.insert("end", f"You: {msg}\n\n")
        self.txt_pdf_chat_history.configure(state="disabled")
        
        def task():
            res = pdf_app.ask_anything(msg, [])
            self.after(10, lambda: self.append_pdf_chat_response(res))
        threading.Thread(target=task, daemon=True).start()

    def append_pdf_chat_response(self, text):
        self.txt_pdf_chat_history.configure(state="normal")
        self.txt_pdf_chat_history.insert("end", f"Aura PDF: {text}\n\n")
        self.txt_pdf_chat_history.yview_moveto(1.0)
        self.txt_pdf_chat_history.configure(state="disabled")

    # ==========================================
    # PERIODIC GUI UPDATE LOOP
    # ==========================================
    def update_gui_loop(self):
        # Update connection statuses periodically from cached background flags
        db_connected = self.db_connected
        self.lbl_db_status.configure(
            text="MongoDB: Connected" if db_connected else "MongoDB: Offline (localhost:27017)",
            text_color="#00ff66" if db_connected else "#ff0066"
        )
        
        ollama_connected = self.ollama_connected
        self.lbl_ollama_status.configure(
            text="Ollama LLM: Connected" if ollama_connected else "Ollama LLM: Standby Fallback",
            text_color="#00ff66" if ollama_connected else "#a1a1b4"
        )
        
        # Update calibration counts display if in calibration screen
        if self.active_tab == "calibration":
            self.update_calibration_sample_counts()
            
        # Real-time Video frame and Metrics fetching
        if self.vision_running:
            # Update webcam preview panel
            frame_img = self.vision.get_latest_frame(target_width=420, target_height=310)
            if frame_img:
                img_tk = ctk.CTkImage(light_image=frame_img, dark_image=frame_img, size=(420, 310))
                self.video_label.configure(image=img_tk, text="")
                self.video_label.image = img_tk
                
            # Update HUD Metrics Panel
            m = self.vision.latest_metrics
            current_state = m["emotion"]
            confidence = m["confidence"]
            focus_val = m["focus_score"]
            blinks = m["blink_count"]
            fatigue = m["fatigue_alert"]
            
            # Text coloring based on active emotion
            text_colors = {
                "Focused": "#00f3ff", # Cyan
                "Happy": "#00ff66",   # Green
                "Stressed": "#ff0066",# Pink-Red
                "Tired": "#ffaa00"    # Orange
            }
            self.lbl_emotion_value.configure(
                text=f"{current_state} (Confidence: {confidence * 100:.0f}%)",
                text_color=text_colors.get(current_state, "#ffffff")
            )
            
            # Progress bar set (value: 0.0 - 1.0)
            self.focus_bar.set(focus_val / 100.0)
            self.lbl_focus_text.configure(text=f"Focus & Attention Index: {focus_val:.0f}%")
            
            # Blink count
            self.lbl_blink_hud.configure(text=f"Session Blinks (Active / 10s): {blinks}")
            
            # Fatigue indicator
            if fatigue:
                self.lbl_fatigue_alert.configure(text="Fatigue Watchdog: SLEEP / FATIGUE ALERT!", text_color="#ffffff")
                self.fatigue_card.configure(fg_color="#550011") # red alert background
            else:
                self.lbl_fatigue_alert.configure(text="Fatigue Watchdog: Normal", text_color="#00ff66")
                self.fatigue_card.configure(fg_color="#242435")
                
        # Re-run loop in 40ms (~25 updates/sec)
        self.after(40, self.update_gui_loop)

    def _connection_check_loop(self):
        while True:
            # Check DB
            db_ok = self.db.check_connection()
            
            # Check Ollama
            ollama_ok = self.ai.check_connection()
            
            # Update cache flags
            self.db_connected = db_ok
            self.ollama_connected = ollama_ok
            
            # If DB just connected and user_id is None, resolve it
            if db_ok and self.user_id is None:
                self.user_id = self.db.get_or_create_user("student_final_project")
                if self.vision:
                    self.vision.user_id = self.user_id
                    # Attempt to retrain classifier since we now have the user_id
                    self.vision.retrain_classifier()
            
            # Wait 5 seconds before next check
            time.sleep(5.0)
        
    def on_closing(self):
        # Stop vision thread gracefully on exit
        if self.vision_running:
            self.vision.stop()
        self.destroy()

if __name__ == "__main__":
    app = AuraVisionApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
