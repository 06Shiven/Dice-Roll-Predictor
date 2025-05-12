import tkinter as tk
import random
import pandas as pd
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

HISTORY_FILE = "roll_history.json"

# === Save/Load Roll History ===
def save_history(recent, log):
    with open(HISTORY_FILE, 'w') as f:
        json.dump({"recent": recent, "log": log}, f)

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):  # old format fallback
                recent = data[-3:]
                log = [{"roll": r} for r in data]
                return recent, log
            return data.get("recent", [random.randint(1, 6) for _ in range(3)]), data.get("log", [])
    recent = [random.randint(1, 6) for _ in range(3)]
    return recent, []

# === Generate Fake Dice Data ===
def generate_data(num_samples=1000, history_length=3):
    X, y = [], []
    rolls = [random.randint(1, 6) for _ in range(num_samples + history_length)]
    for i in range(num_samples):
        X.append(rolls[i:i+history_length])
        y.append(rolls[i+history_length])
    return pd.DataFrame(X, columns=[f'roll_{i+1}' for i in range(history_length)]), pd.Series(y)

X, y = generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# === GUI App ===
class DiceApp:
    def __init__(self, master):
        self.master = master
        master.title("🎲 Dice Roll Predictor")
        master.geometry("420x400")
        master.configure(bg="#1e1e1e")

        self.roll_history, self.roll_log = load_history()
        self.last_prediction = None  # stores last prediction before rolling

        self.history_label = tk.Label(master, text="", font=("Courier", 16), bg="#1e1e1e", fg="white")
        self.history_label.pack(pady=20)

        self.roll_button = tk.Button(master, text="Roll Dice", command=self.roll_dice, font=("Arial", 14), bg="#333", fg="white")
        self.roll_button.pack(pady=10)

        self.predict_button = tk.Button(master, text="Predict Next Roll", command=self.predict, font=("Arial", 14), bg="#007acc", fg="white")
        self.predict_button.pack(pady=10)

        self.history_button = tk.Button(master, text="View Roll Log", command=self.show_roll_log, font=("Arial", 12), bg="#555", fg="white")
        self.history_button.pack(pady=10)

        self.result_label = tk.Label(master, text="", font=("Courier", 18, "bold"), bg="#1e1e1e", fg="#00ffcc")
        self.result_label.pack(pady=20)

        self.update_history_label()

    def update_history_label(self):
        self.history_label.config(text=f"Last Rolls: {self.roll_history}")

    def roll_dice(self):
        new_roll = random.randint(1, 6)
        self.roll_history.append(new_roll)
        self.roll_history = self.roll_history[-3:]

        log_entry = {"roll": int(new_roll)}

        if self.last_prediction is not None:
            log_entry["predicted"] = int(self.last_prediction)
            log_entry["correct"] = (self.last_prediction == new_roll)
            self.last_prediction = None

        self.roll_log.append(log_entry)
        save_history(self.roll_history, self.roll_log)

        self.update_history_label()
        self.result_label.config(text=f"You rolled: {new_roll}")

    def predict(self):
        input_df = pd.DataFrame([self.roll_history], columns=[f'roll_{i+1}' for i in range(3)])
        prediction = int(model.predict(input_df)[0])  # convert to native int
        self.last_prediction = prediction
        self.result_label.config(text=f"AI predicts: {prediction}")

    def show_roll_log(self):
        log_window = tk.Toplevel(self.master)
        log_window.title("Roll History")
        log_window.geometry("350x500")
        log_window.configure(bg="#1e1e1e")
        text = tk.Text(log_window, wrap="word", bg="#121212", fg="white", font=("Courier", 11))

        for entry in self.roll_log:
            if isinstance(entry, dict):
                line = f"Roll: {entry['roll']}"
                if "predicted" in entry:
                    line += f" | Predicted: {entry['predicted']} {'✅' if entry['correct'] else '❌'}"
            else:
                # fallback for old plain-int entries
                line = f"Roll: {entry}"
            line += "\n"
            text.insert("end", line)

        text.config(state="disabled")
        text.pack(expand=True, fill="both")

# === Run ===
if __name__ == "__main__":
    root = tk.Tk()
    app = DiceApp(root)
    root.mainloop()
