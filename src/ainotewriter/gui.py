from __future__ import annotations

import json
import os
import threading
import tkinter as tk
from dataclasses import asdict
from tkinter import messagebox, ttk

from .config import AppConfig
from .service import CommunityNoteWriterService, save_recent_notes, save_summary


class NoteWriterApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("X Community Notes AI Writer")
        self.root.geometry("980x700")

        self.config = AppConfig.from_env()
        self.service = CommunityNoteWriterService(self.config)

        self.num_posts = tk.IntVar(value=self.config.default_num_posts)
        self.test_mode = tk.BooleanVar(value=self.config.default_test_mode)
        self.submit_notes = tk.BooleanVar(value=self.config.default_submit_notes)
        self.evaluate = tk.BooleanVar(value=self.config.default_evaluate_before_submit)
        self.min_score = tk.DoubleVar(value=self.config.default_min_claim_opinion_score)

        self.last_output_path: str | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        ctrl = ttk.Frame(self.root, padding=12)
        ctrl.pack(fill=tk.X)

        ttk.Label(ctrl, text="Num posts").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(ctrl, textvariable=self.num_posts, width=8).grid(row=0, column=1, padx=6)

        ttk.Checkbutton(ctrl, text="Test mode", variable=self.test_mode).grid(row=1, column=0, sticky=tk.W)
        ttk.Checkbutton(ctrl, text="Submit notes", variable=self.submit_notes).grid(row=1, column=1, sticky=tk.W)
        ttk.Checkbutton(ctrl, text="Evaluate before submit", variable=self.evaluate).grid(row=1, column=2, sticky=tk.W)

        ttk.Label(ctrl, text="Min claim/opinion score").grid(row=1, column=3, sticky=tk.W)
        ttk.Entry(ctrl, textvariable=self.min_score, width=8).grid(row=1, column=4, padx=6)

        btns = ttk.Frame(self.root, padding=(12, 0, 12, 8))
        btns.pack(fill=tk.X)
        ttk.Button(btns, text="Run writer", command=self.run_writer).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btns, text="Fetch notes_written", command=self.fetch_notes).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btns, text="Open last JSON", command=self.open_last_json).pack(side=tk.LEFT)

        self.log = tk.Text(self.root, wrap=tk.WORD, font=("Consolas", 10))
        self.log.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

    def _append(self, text: str) -> None:
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)

    def run_writer(self) -> None:
        def _task():
            try:
                self._append("Running...")
                summary = self.service.run_once(
                    num_posts=self.num_posts.get(),
                    test_mode=self.test_mode.get(),
                    submit_notes=self.submit_notes.get(),
                    evaluate_before_submit=self.evaluate.get(),
                    min_claim_opinion_score=self.min_score.get(),
                    progress_callback=self._append,
                )
                path = save_summary(summary)
                self.last_output_path = str(path)
                self._append(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
                self._append(f"Saved: {path}")
            except Exception as ex:
                self._append(f"ERROR: {ex}")
                messagebox.showerror("Error", str(ex))

        threading.Thread(target=_task, daemon=True).start()

    def fetch_notes(self) -> None:
        def _task():
            try:
                self._append("Fetching notes_written...")
                notes = self.service.fetch_recent_notes(max_results=20, test_mode=self.test_mode.get())
                path = save_recent_notes(notes)
                self.last_output_path = str(path)
                self._append(json.dumps(notes, ensure_ascii=False, indent=2))
                self._append(f"Saved: {path}")
            except Exception as ex:
                self._append(f"ERROR: {ex}")
                messagebox.showerror("Error", str(ex))

        threading.Thread(target=_task, daemon=True).start()

    def open_last_json(self) -> None:
        if not self.last_output_path:
            messagebox.showinfo("Info", "No output file yet.")
            return
        if not os.path.exists(self.last_output_path):
            messagebox.showwarning("Warning", "Output file does not exist anymore.")
            return
        os.startfile(self.last_output_path)  # type: ignore[attr-defined]


def main() -> None:
    root = tk.Tk()
    app = NoteWriterApp(root)
    app._append("Ready. Configure and click 'Run writer'.")
    root.mainloop()


if __name__ == "__main__":
    main()
