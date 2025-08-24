# app.py
import threading
import random
import time
from collections import Counter, defaultdict, deque
from datetime import datetime

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

# Charts + Graph
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import networkx as nx

# Optional ML (auto-enables if installed)
USE_ML = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    USE_ML = True
except Exception:
    USE_ML = False


# =========================
#   DSA: TRIE (Prefix Tree)
# =========================
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.label = None  # category

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, phrase, label=None):
        node = self.root
        for ch in phrase:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True
        node.label = label

    def find_all_in(self, text):
        """Find all phrases present in text. Returns list[(phrase,label)]."""
        found = []
        n = len(text)
        for i in range(n):
            node = self.root
            j = i
            current = []
            while j < n and text[j] in node.children:
                node = node.children[text[j]]
                current.append(text[j])
                if node.is_end:
                    found.append(("".join(current), node.label))
                j += 1
        return found


# =========================
#   RULES / DATA
# =========================
RULES = {
    "buy now": "ad",
    "click here": "ad",
    "limited offer": "ad",
    "subscribe": "ad",
    "win money": "scam",
    "lottery": "scam",
    "prize": "scam",
    "free gift": "scam",
    "urgent action": "phishing",
    "verify account": "phishing",
}

CATEGORY_NAME = {
    "ad": "Advertisement",
    "scam": "Scam",
    "phishing": "Phishing",
    "iot": "IoT Alert",
    "safe": "Safe",
}

# Simulated stream of messages (clean + spammy)
CLEAN_MESSAGES = [
    "Temperature is 30 C", "Water level normal", "Motion detected in hallway",
    "Humidity is 54 percent", "Battery low alert", "Door closed", "Light turned on",
    "System update available", "Device connected to gateway"
]
SPAM_MESSAGES = [
    "Buy now and win money", "Click here to claim your prize",
    "Limited offer subscribe today", "Urgent action verify account",
    "Free gift lottery winner"
]

DEVICES = ["Sensor1", "Sensor2", "Camera1", "Light1", "Thermostat", "Gateway"]
EDGES = [
    ("Sensor1", "Gateway"), ("Sensor2", "Gateway"),
    ("Camera1", "Gateway"), ("Light1", "Gateway"),
    ("Thermostat", "Gateway")
]


# =========================
#   OPTIONAL ML CLASSIFIER
# =========================
class SimpleMLSpam:
    def __init__(self):
        self.ready = False
        if not USE_ML:
            return
        # Tiny offline training set (expand if you wish)
        X_train = [
            "buy now limited offer", "click here to subscribe",
            "win money lottery prize", "urgent action verify account", "free gift claim now",
            "temperature is 28 degrees", "water level normal", "motion detected living room",
            "battery low alert", "humidity 40 percent normal",
        ]
        y_train = [1,1,1,1,1, 0,0,0,0,0]  # 1=spam, 0=safe
        self.vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
        Xv = self.vec.fit_transform([t.lower() for t in X_train])
        self.clf = LogisticRegression(max_iter=300)
        self.clf.fit(Xv, y_train)
        self.ready = True

    def predict_spam(self, text: str) -> bool:
        if not self.ready:
            return False
        Xv = self.vec.transform([text.lower()])
        return bool(self.clf.predict(Xv)[0])


# =========================
#   CORE SPAM ENGINE
# =========================
class SpamEngine:
    def __init__(self):
        # DSA: Trie rules
        self.trie = Trie()
        for phrase, label in RULES.items():
            self.trie.insert(phrase, label)
        # ML
        self.ml = SimpleMLSpam()
        # Stats
        self.stats = Counter()     # counts per category + 'spam'/'safe'
        self.per_device_state = {} # device -> "safe"/"spam"
        # Frequency-based detection: track recent messages per device
        self.recent_msgs = defaultdict(lambda: deque(maxlen=8))  # device -> deque of last messages
        self.repeat_threshold = 3  # if same msg repeats >= threshold in window -> spam

    def add_rule(self, phrase: str, label: str):
        phrase = phrase.strip().lower()
        if not phrase: return
        RULES[phrase] = label
        self.trie.insert(phrase, label)

    def classify(self, device: str, message: str):
        """Returns: is_spam(bool), categories[list], reasons[str list]"""
        text = message.lower().strip()
        reasons = []

        # 1) Trie rule matching
        matches = self.trie.find_all_in(text)
        categories = list({lab for (_, lab) in matches})
        if categories:
            reasons.append("Rule match: " + ", ".join([f"'{p}'→{lab}" for (p, lab) in matches]))

        # 2) Frequency/Repeat detection
        dq = self.recent_msgs[device]
        dq.append(text)
        same_count = sum(1 for m in dq if m == text)
        freq_spam = same_count >= self.repeat_threshold
        if freq_spam:
            reasons.append(f"Repeated message (x{same_count})")

        # 3) Optional ML
        ml_spam = False
        if self.ml.ready:
            ml_spam = self.ml.predict_spam(text)
            if ml_spam:
                reasons.append("ML predicted spam")

        # Final decision
        is_spam = bool(categories) or freq_spam or ml_spam

        # Normalize categories
        final_categories = categories[:]
        if not final_categories and is_spam:
            final_categories = ["ad"]  # generic spam bucket if ML/freq only
        if not is_spam:
            final_categories = ["iot"]

        # Update stats
        if is_spam:
            self.stats["spam"] += 1
            for c in final_categories:
                self.stats[c] += 1
        else:
            self.stats["safe"] += 1
            self.stats["iot"] += 1

        self.per_device_state[device] = "spam" if is_spam else "safe"
        return is_spam, final_categories, reasons

    def report_counts(self):
        return dict(self.stats)

    def reset(self):
        self.stats.clear()
        self.per_device_state.clear()
        self.recent_msgs.clear()


# =========================
#   GUI APP
# =========================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("IoT Spam Control — Real-Time Visualization (DSA + ML)")
        self.root.geometry("1100x700")

        self.engine = SpamEngine()
        self.running = False
        self.sim_thread = None
        self.lock = threading.Lock()

        # --- Layout: Left (graph) | Right (controls + log)
        container = ttk.Frame(root)
        container.pack(fill="both", expand=True, padx=8, pady=8)
        container.columnconfigure(0, weight=3)
        container.columnconfigure(1, weight=2)
        container.rowconfigure(0, weight=1)

        # Graph panel
        graph_frame = ttk.LabelFrame(container, text="IoT Network")
        graph_frame.grid(row=0, column=0, sticky="nsew", padx=(0,8))
        graph_frame.rowconfigure(0, weight=1)
        graph_frame.columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(6.5,5.2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Controls + Log
        right_frame = ttk.LabelFrame(container, text="Controls & Log")
        right_frame.grid(row=0, column=1, sticky="nsew")
        right_frame.rowconfigure(2, weight=1)
        right_frame.columnconfigure(0, weight=1)

        # Buttons
        btns = ttk.Frame(right_frame)
        btns.grid(row=0, column=0, sticky="ew", padx=6, pady=6)
        for i in range(4): btns.columnconfigure(i, weight=1)

        self.start_btn = ttk.Button(btns, text="Start Simulation", command=self.start_sim)
        self.stop_btn  = ttk.Button(btns, text="Stop", command=self.stop_sim)
        self.attack_btn= ttk.Button(btns, text="Attack Simulation", command=self.attack_sim)
        self.clear_btn = ttk.Button(btns, text="Clear/Reset", command=self.reset_all)
        self.start_btn.grid(row=0, column=0, padx=3, pady=3, sticky="ew")
        self.stop_btn.grid( row=0, column=1, padx=3, pady=3, sticky="ew")
        self.attack_btn.grid(row=0, column=2, padx=3, pady=3, sticky="ew")
        self.clear_btn.grid(row=0, column=3, padx=3, pady=3, sticky="ew")

        # Charts + Add rule
        btns2 = ttk.Frame(right_frame)
        btns2.grid(row=1, column=0, sticky="ew", padx=6, pady=(0,6))
        for i in range(3): btns2.columnconfigure(i, weight=1)

        self.report_btn = ttk.Button(btns2, text="Show Charts", command=self.show_charts)
        self.addrule_btn = ttk.Button(btns2, text="Add Custom Rule", command=self.add_rule_dialog)
        self.ml_label = ttk.Label(btns2, text=("ML: ON" if USE_ML else "ML: OFF"), anchor="center")
        self.report_btn.grid(row=0, column=0, padx=3, pady=3, sticky="ew")
        self.addrule_btn.grid(row=0, column=1, padx=3, pady=3, sticky="ew")
        self.ml_label.grid(row=0, column=2, padx=3, pady=3, sticky="ew")

        # Log
        self.log = tk.Text(right_frame, height=20, wrap="word")
        self.log.grid(row=2, column=0, sticky="nsew", padx=6, pady=6)
        self.log.config(state="disabled")

        # Build network graph
        self.G = nx.Graph()
        self.G.add_nodes_from(DEVICES)
        self.G.add_edges_from(EDGES)
        self.pos = nx.spring_layout(self.G, seed=42)  # fixed layout

        # Device state
        self.device_state = {d: "safe" for d in DEVICES}
        self.device_state["Gateway"] = "safe"

        # Initial draw
        self.draw_graph()

        # Graceful close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # --- Graph drawing
    def draw_graph(self):
        self.ax.clear()
        colors = ["red" if self.engine.per_device_state.get(n, "safe") == "spam" else "green" for n in self.G.nodes]
        nx.draw(self.G, pos=self.pos, ax=self.ax, with_labels=True,
                node_color=colors, node_size=1500, font_size=9, font_color="white", width=1.5)
        self.ax.set_title("Green = Safe | Red = Spam")
        self.canvas.draw_idle()

    # --- Logging helper
    def append_log(self, line: str):
        self.log.config(state="normal")
        self.log.insert("end", f"[{datetime.now().strftime('%H:%M:%S')}] {line}\n")
        self.log.see("end")
        self.log.config(state="disabled")

    # --- Simulation loop (thread)
    def sim_loop(self):
        while self.running:
            device = random.choice([d for d in DEVICES if d != "Gateway"])
            # mix clean/spam with some probability
            if random.random() < 0.45:
                msg = random.choice(SPAM_MESSAGES)
            else:
                msg = random.choice(CLEAN_MESSAGES)

            is_spam, cats, reasons = self.engine.classify(device, msg)

            if is_spam:
                cat_names = ", ".join(CATEGORY_NAME.get(c, c) for c in cats)
                self.root.after(0, self.append_log, f"🚨 {device} → SPAM ({cat_names}) | msg: {msg}")
                if reasons:
                    self.root.after(0, self.append_log, f"   reasons: {', '.join(reasons)}")
            else:
                self.root.after(0, self.append_log, f"✅ {device} → SAFE | msg: {msg}")

            self.root.after(0, self.draw_graph)
            time.sleep(1.6)

    # --- Controls
    def start_sim(self):
        if self.running: return
        self.running = True
        self.append_log("▶ Simulation started.")
        self.sim_thread = threading.Thread(target=self.sim_loop, daemon=True)
        self.sim_thread.start()

    def stop_sim(self):
        if not self.running: return
        self.running = False
        self.append_log("⏸ Simulation stopped.")

    def attack_sim(self):
        """Burst of spam from multiple devices (botnet-like)."""
        attackers = random.sample([d for d in DEVICES if d != "Gateway"], k=min(3, len(DEVICES)-1))
        burst = random.choice(SPAM_MESSAGES)
        self.append_log(f"⚠ Attack started by: {', '.join(attackers)} | payload: {burst}")
        for dev in attackers:
            is_spam, cats, reasons = self.engine.classify(dev, burst)
        self.draw_graph()

    def reset_all(self):
        self.stop_sim()
        self.engine.reset()
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")
        self.append_log("🔄 Reset performed. Stats cleared, devices set to SAFE.")
        self.draw_graph()

    def add_rule_dialog(self):
        phrase = simpledialog.askstring("Add Rule", "Enter spam phrase (e.g., 'claim prize now'):")
        if not phrase: return
        cat = simpledialog.askstring("Add Rule", "Enter category (ad / scam / phishing):", initialvalue="ad")
        if not cat: return
        cat = cat.strip().lower()
        if cat not in {"ad","scam","phishing"}:
            messagebox.showerror("Error", "Category must be one of: ad, scam, phishing")
            return
        self.engine.add_rule(phrase.lower(), cat)
        self.append_log(f"➕ Rule added: '{phrase.lower()}' → {cat}")

    def show_charts(self):
        stats = self.engine.report_counts()
        spam = stats.get("spam", 0)
        safe = stats.get("safe", 0)
        cats = ["Advertisement", "Scam", "Phishing", "IoT Alert"]
        vals = [stats.get("ad",0), stats.get("scam",0), stats.get("phishing",0), stats.get("iot",0)]

        fig = Figure(figsize=(8.5,4), dpi=100)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # Pie (Spam vs Safe)
        total = spam + safe
        if total == 0:
            ax1.text(0.5,0.5,"No data yet", ha="center", va="center")
        else:
            ax1.pie([spam, safe], labels=["Spam","Safe"], autopct="%1.0f%%", startangle=140)
        ax1.set_title("Spam vs Safe")

        # Bar (Category breakdown)
        ax2.bar(cats, vals)
        ax2.set_title("Category Breakdown")
        ax2.set_xticklabels(cats, rotation=20, ha="right")

        # Show in new Tk window
        win = tk.Toplevel(self.root)
        win.title("Charts")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()

    def on_close(self):
        self.stop_sim()
        self.root.after(200, self.root.destroy)


def main():
    root = tk.Tk()
    # prettier theme if available
    try:
        from tkinter import font as tkfont
        style = ttk.Style()
        style.theme_use("clam")
    except Exception:
        pass
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
