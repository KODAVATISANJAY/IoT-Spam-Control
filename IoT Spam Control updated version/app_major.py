import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
import sqlite3
from datetime import datetime
from collections import deque, Counter

# ---- OPTIONAL ML (auto-detect) ----
USE_ML = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    USE_ML = True
except Exception:
    USE_ML = False

# ---- CHARTS ----
import matplotlib.pyplot as plt

# =========================
#   DSA: TRIE (Prefix Tree)
# =========================
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.label = None  # category/label for this phrase

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
        """Return list of (matched_phrase, label) found in text."""
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
#   DATA / RULES
# =========================
# phrase -> category
RULES = {
    "buy now": "ad",
    "click here": "ad",
    "subscribe": "ad",
    "limited offer": "ad",
    "win money": "scam",
    "lottery": "scam",
    "prize": "scam",
    "free gift": "scam",
    "urgent action": "phishing",
    "verify account": "phishing",
}

CATEGORIES_FRIENDLY = {
    "ad": "Advertisement",
    "scam": "Scam",
    "phishing": "Phishing",
    "iot": "IoT Alert",
    "safe": "Safe",
}

# Example IoT messages to simulate incoming stream
SIM_STREAM = deque([
    "Temperature is 30°C",
    "Buy now your free gift",
    "Water level low in tank",
    "Click here to claim your lottery prize",
    "Motion detected in living room",
    "Urgent action required: verify account",
    "Humidity is 54 percent",
])

# =========================
#   OPTIONAL ML CLASSIFIER
# =========================
class SimpleMLSpam:
    def __init__(self):
        self.ready = False
        if not USE_ML:
            return

        # Tiny built-in training set so it works offline.
        # Feel free to expand with your own labeled lines.
        X_train = [
            "buy now limited offer",
            "click here to subscribe",
            "win money lottery prize",
            "urgent action verify account",
            "free gift claim now",
            "temperature is 28 degrees",
            "water level normal",
            "motion detected at kitchen door",
            "device battery low alert",
            "humidity 40 percent normal",
        ]
        y_train = [
            1, 1, 1, 1, 1,    # spam
            0, 0, 0, 0, 0     # safe
        ]

        self.vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
        Xv = self.vec.fit_transform([t.lower() for t in X_train])
        self.clf = LogisticRegression(max_iter=200)
        self.clf.fit(Xv, y_train)
        self.ready = True

    def predict_spam(self, text: str) -> bool:
        if not self.ready:
            return False
        Xv = self.vec.transform([text.lower()])
        pred = self.clf.predict(Xv)[0]
        return bool(pred)

# =========================
#   DATABASE (SQLite)
# =========================
DB_NAME = "iot_spam_history.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            message TEXT,
            decision TEXT,      -- 'spam' or 'safe'
            categories TEXT     -- comma-separated categories found
        )
    """)
    conn.commit()
    conn.close()

def log_message(message, decision, categories):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO messages (timestamp, message, decision, categories) VALUES (?, ?, ?, ?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), message, decision, ",".join(categories) if categories else "")
    )
    conn.commit()
    conn.close()

def fetch_history(limit=200):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT timestamp, message, decision, categories FROM messages ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return rows

# =========================
#   CORE ENGINE
# =========================
class SpamEngine:
    def __init__(self):
        # Build Trie from RULES
        self.trie = Trie()
        for phrase, label in RULES.items():
            self.trie.insert(phrase, label)
        # ML
        self.ml = SimpleMLSpam()
        self.stats = Counter()  # counts per category + 'spam'/'safe'

    def classify(self, msg: str):
        text = msg.lower().strip()

        # 1) DSA (Trie) rules
        matches = self.trie.find_all_in(text)  # list of (phrase, label)
        categories = list({lab for (_, lab) in matches})  # unique cats
        spam_by_rules = len(categories) > 0

        # 2) Optional ML (overrides or augments)
        spam_by_ml = False
        if self.ml.ready:
            spam_by_ml = self.ml.predict_spam(text)

        # Decision fusion:
        # If either rules OR ML says spam => spam
        is_spam = spam_by_rules or spam_by_ml

        # Assign top-level category list
        final_categories = categories[:]
        if not final_categories and is_spam:
            # ML-only spam – put into 'ad' as generic or 'phishing' heuristic
            final_categories = ["ad"]

        if not is_spam:
            final_categories = ["iot"]  # treat as IoT alert/safe message

        # Update stats
        if is_spam:
            self.stats["spam"] += 1
            for c in final_categories:
                self.stats[c] += 1
        else:
            self.stats["safe"] += 1
            self.stats["iot"] += 1

        return is_spam, final_categories, matches

    def report_counts(self):
        return dict(self.stats)

# =========================
#   GUI (Tkinter)
# =========================
class App:
    def __init__(self, root):
        self.root = root
        self.engine = SpamEngine()
        init_db()

        root.title("IoT Spam Control & Smart Monitoring")
        root.geometry("800x600")
        root.config(bg="#f7f7fb")

        # Title
        title = tk.Label(root, text="🛰️  IoT Spam Control & Smart Monitoring",
                         font=("Segoe UI", 16, "bold"), bg="#2d2f36", fg="white", pady=12)
        title.pack(fill="x")

        # Input area
        frm_in = tk.Frame(root, bg="#f7f7fb")
        frm_in.pack(pady=10, fill="x")

        tk.Label(frm_in, text="Enter / Simulate IoT Message:",
                 font=("Segoe UI", 11), bg="#f7f7fb").pack(anchor="w", padx=20)

        self.entry = scrolledtext.ScrolledText(frm_in, width=90, height=4, font=("Consolas", 11))
        self.entry.pack(padx=20, pady=6, fill="x")

        # Buttons
        frm_btn = tk.Frame(root, bg="#f7f7fb")
        frm_btn.pack(pady=6)

        self.btn_check = ttk.Button(frm_btn, text="Check Message", command=self.check_message)
        self.btn_check.grid(row=0, column=0, padx=6)

        self.btn_clear = ttk.Button(frm_btn, text="Clear", command=self.clear_message)
        self.btn_clear.grid(row=0, column=1, padx=6)

        self.btn_sim = ttk.Button(frm_btn, text="Simulate Next IoT Message", command=self.simulate_message)
        self.btn_sim.grid(row=0, column=2, padx=6)

        self.btn_report = ttk.Button(frm_btn, text="Show Report (Chart)", command=self.show_report)
        self.btn_report.grid(row=0, column=3, padx=6)

        self.btn_history = ttk.Button(frm_btn, text="View History", command=self.view_history)
        self.btn_history.grid(row=0, column=4, padx=6)

        # Result panel
        frm_res = tk.LabelFrame(root, text="Result", font=("Segoe UI", 11, "bold"))
        frm_res.pack(padx=20, pady=10, fill="both")

        self.lbl_decision = tk.Label(frm_res, text="Result will appear here...",
                                     font=("Segoe UI", 12), fg="blue", wraplength=740, justify="left")
        self.lbl_decision.pack(anchor="w", padx=10, pady=8)

        self.lbl_matches = tk.Label(frm_res, text="", font=("Segoe UI", 10), fg="#444",
                                    wraplength=740, justify="left")
        self.lbl_matches.pack(anchor="w", padx=10, pady=2)

        # Status bar
        self.status = tk.Label(root, text="Ready", anchor="w", bg="#eee", fg="#333", padx=8)
        self.status.pack(side="bottom", fill="x")

        # Style
        style = ttk.Style()
        style.theme_use("clam")

    def check_message(self):
        msg = self.entry.get("1.0", tk.END).strip()
        if not msg:
            messagebox.showinfo("Info", "Please enter a message.")
            return

        is_spam, categories, matches = self.engine.classify(msg)

        if is_spam:
            cat_names = ", ".join(CATEGORIES_FRIENDLY.get(c, c) for c in categories)
            self.lbl_decision.config(text=f"🚨 SPAM detected — Categories: {cat_names}", fg="white", bg="#c0392b")
        else:
            self.lbl_decision.config(text="✅ SAFE (IoT alert/data)", fg="white", bg="#27ae60")

        if matches:
            self.lbl_matches.config(
                text="Matched phrases: " + ", ".join([f"“{p}”→{CATEGORIES_FRIENDLY.get(l, l)}" for (p, l) in matches])
            )
        else:
            self.lbl_matches.config(text="No rule-based matches. (ML may have classified it if enabled)")

        # Log to DB
        log_message(msg, "spam" if is_spam else "safe", categories)
        self.status.config(text="Saved to history.")

    def clear_message(self):
        self.entry.delete("1.0", tk.END)
        self.lbl_decision.config(text="Result will appear here...", fg="blue", bg=self.root.cget("bg"))
        self.lbl_matches.config(text="")
        self.status.config(text="Cleared.")

    def simulate_message(self):
        if not SIM_STREAM:
            messagebox.showinfo("Info", "No more simulated messages in the queue.")
            return
        msg = SIM_STREAM.popleft()
        self.entry.delete("1.0", tk.END)
        self.entry.insert("1.0", msg)
        self.status.config(text="Loaded a simulated IoT message. Click 'Check Message'.")

    def show_report(self):
        stats = self.engine.report_counts()
        # Build counts
        spam = stats.get("spam", 0)
        safe = stats.get("safe", 0)

        # Category breakdown (ad/scam/phishing/iot)
        bars_labels = []
        bars_values = []
        for key in ["ad", "scam", "phishing", "iot"]:
            val = stats.get(key, 0)
            bars_labels.append(CATEGORIES_FRIENDLY.get(key, key))
            bars_values.append(val)

        # Plot
        plt.figure(figsize=(7,4))
        plt.subplot(1,2,1)
        plt.title("Spam vs Safe")
        plt.bar(["Spam", "Safe"], [spam, safe])

        plt.subplot(1,2,2)
        plt.title("Category Breakdown")
        plt.bar(bars_labels, bars_values)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.show()

    def view_history(self):
        rows = fetch_history(200)
        if not rows:
            messagebox.showinfo("History", "No records yet.")
            return

        win = tk.Toplevel(self.root)
        win.title("Message History (SQLite)")
        win.geometry("820x400")

        cols = ("Time", "Message", "Decision", "Categories")
        tree = ttk.Treeview(win, columns=cols, show="headings")
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=200 if c != "Message" else 360, anchor="w")
        tree.pack(fill="both", expand=True)

        for ts, msg, dec, cats in rows:
            tree.insert("", "end", values=(ts, msg[:200], dec.upper(), cats))

        ttk.Button(win, text="Close", command=win.destroy).pack(pady=6)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
