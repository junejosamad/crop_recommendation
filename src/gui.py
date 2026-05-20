from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import joblib
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from models import train_all
from utils import MODEL_DIR, RESULTS_DIR, agronomic_cluster_guidance, yield_confidence_bounds


DEFAULT_INPUTS = {
    "N": 70,
    "P": 45,
    "K": 40,
    "temperature": 25,
    "humidity": 70,
    "ph": 6.5,
    "rainfall": 120,
}


class AgricultureApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Smart Agriculture Decision Support System")
        self.geometry("1180x760")
        self.minsize(980, 680)
        self.bundle = self._load_bundle()
        self.entries: dict[str, ttk.Entry] = {}
        self.result_var = tk.StringVar(value="Enter soil and climate readings, then run inference.")
        self._build_layout()

    def _load_bundle(self) -> dict:
        path = MODEL_DIR / "agri_ai_bundle.joblib"
        if not path.exists():
            train_all()
        return joblib.load(path)

    def _build_layout(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=16)
        left.grid(row=0, column=0, sticky="ns")
        right = ttk.Frame(self, padding=(0, 16, 16, 16))
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        ttk.Label(left, text="Input Parameters", font=("Segoe UI", 14, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 12)
        )

        for row, (name, value) in enumerate(DEFAULT_INPUTS.items(), start=1):
            ttk.Label(left, text=name).grid(row=row, column=0, sticky="w", pady=5)
            entry = ttk.Entry(left, width=18)
            entry.insert(0, str(value))
            entry.grid(row=row, column=1, sticky="ew", pady=5)
            self.entries[name] = entry

        ttk.Button(left, text="Run Inference", command=self.run_inference).grid(
            row=9, column=0, columnspan=2, sticky="ew", pady=(16, 6)
        )
        ttk.Button(left, text="Retrain Models", command=self.retrain_models).grid(
            row=10, column=0, columnspan=2, sticky="ew"
        )

        ttk.Label(left, text="Integrated Output", font=("Segoe UI", 12, "bold")).grid(
            row=11, column=0, columnspan=2, sticky="w", pady=(20, 8)
        )
        ttk.Label(left, textvariable=self.result_var, wraplength=310, justify="left").grid(
            row=12, column=0, columnspan=2, sticky="nw"
        )

        ttk.Label(right, text="Model Visualizations", font=("Segoe UI", 14, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 12)
        )
        notebook = ttk.Notebook(right)
        notebook.grid(row=1, column=0, sticky="nsew")
        for title, filename in [
            ("Feature Importance", "feature_importance.png"),
            ("Cluster Scatter", "cluster_scatter.png"),
            ("Residual Plot", "residual_plot.png"),
        ]:
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=title)
            self._add_image_plot(frame, RESULTS_DIR / filename)

    def _add_image_plot(self, parent: ttk.Frame, image_path: Path) -> None:
        fig = Figure(figsize=(7.8, 5.1), dpi=100)
        ax = fig.add_subplot(111)
        ax.axis("off")
        if image_path.exists():
            image = plt_read_image(str(image_path))
            ax.imshow(image)
        else:
            ax.text(0.5, 0.5, "Plot not available. Retrain models.", ha="center", va="center")
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _read_input_frame(self) -> pd.DataFrame:
        values = {}
        for name, entry in self.entries.items():
            values[name] = float(entry.get())
        return pd.DataFrame([values], columns=self.bundle["feature_columns"])

    def run_inference(self) -> None:
        try:
            sample = self._read_input_frame()
            crop_code = self.bundle["crop_model"].predict(sample)[0]
            crop = self.bundle["label_encoder"].inverse_transform([crop_code])[0]
            cluster = int(self.bundle["knn_cluster_model"].predict(sample)[0])
            yield_prediction = float(self.bundle["yield_model"].predict(sample)[0])
            low, high = yield_confidence_bounds(yield_prediction, self.bundle["yield_rmse"])
            self.result_var.set(
                f"Recommended crop: {crop}\n"
                f"Soil cluster: {cluster}\n"
                f"Guidance: {agronomic_cluster_guidance(cluster)}\n"
                f"Predicted yield index: {yield_prediction:.2f}\n"
                f"Approx. 95% bounds: {low:.2f} to {high:.2f}"
            )
        except ValueError:
            messagebox.showerror("Invalid input", "All inputs must be numeric.")

    def retrain_models(self) -> None:
        train_all()
        self.bundle = self._load_bundle()
        messagebox.showinfo("Training complete", "Models and result plots were regenerated.")


def plt_read_image(path: str):
    import matplotlib.pyplot as plt

    return plt.imread(path)


if __name__ == "__main__":
    app = AgricultureApp()
    app.mainloop()
