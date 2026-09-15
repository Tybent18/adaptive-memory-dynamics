"""One-click desktop laboratory for collecting and exporting evidence."""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from tkinter import (
    BOTH,
    LEFT,
    RIGHT,
    Button,
    DoubleVar,
    Frame,
    IntVar,
    Label,
    StringVar,
    Tk,
    messagebox,
    ttk,
)

from .experiment import run_suite
from .visualization import create_charts

BG = "#07111d"
PANEL = "#0d1b2a"
TEXT = "#d9e7ef"
MUTED = "#7f9bad"
ACCENT = "#32d5a4"


class MemoryLab(Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Adaptive Memory Dynamics — Stage One Laboratory")
        self.geometry("1040x680")
        self.minsize(900, 600)
        self.configure(bg=BG)
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.stop_requested = False
        self.latest_output: Path | None = None
        self.run_settings: tuple[int, int, int] = (5, 24, 16)
        self.seeds = IntVar(value=5)
        self.epochs = IntVar(value=24)
        self.continual_epochs = IntVar(value=16)
        self.status = StringVar(value="Ready to collect reproducible evidence")
        self.progress = DoubleVar(value=0)
        self._build()
        self.after(100, self._poll)

    def _build(self) -> None:
        header = Frame(self, bg=BG, padx=32, pady=24)
        header.pack(fill="x")
        Label(
            header, text="ADAPTIVE MEMORY", fg=ACCENT, bg=BG, font=("Helvetica", 11, "bold")
        ).pack(anchor="w")
        Label(
            header,
            text="Forgetting Dynamics Laboratory",
            fg=TEXT,
            bg=BG,
            font=("Helvetica", 25, "bold"),
        ).pack(anchor="w", pady=(5, 2))
        Label(
            header,
            text="Controlled retention, adaptation, and continual-learning experiments",
            fg=MUTED,
            bg=BG,
            font=("Helvetica", 11),
        ).pack(anchor="w")

        body = Frame(self, bg=BG, padx=32, pady=8)
        body.pack(fill=BOTH, expand=True)
        controls = Frame(body, bg=PANEL, padx=24, pady=24)
        controls.pack(side=LEFT, fill="y", padx=(0, 18))
        Label(
            controls, text="EXPERIMENT CONTROL", fg=TEXT, bg=PANEL, font=("Helvetica", 12, "bold")
        ).pack(anchor="w", pady=(0, 18))
        self._field(controls, "Independent seeds", self.seeds, 2, 20)
        self._field(controls, "Supervised epochs", self.epochs, 4, 100)
        self._field(controls, "Continual epochs / task", self.continual_epochs, 4, 100)
        Button(
            controls,
            text="COLLECT + EXPORT",
            command=self._start,
            bg=ACCENT,
            fg="#032119",
            activebackground="#67e6c0",
            relief="flat",
            font=("Helvetica", 11, "bold"),
            padx=18,
            pady=12,
        ).pack(fill="x", pady=(24, 8))
        Button(
            controls,
            text="STOP AFTER CURRENT RUN",
            command=self._stop,
            bg="#27394a",
            fg=TEXT,
            activebackground="#364d61",
            relief="flat",
            padx=18,
            pady=10,
        ).pack(fill="x", pady=4)
        Button(
            controls,
            text="OPEN LATEST RESULTS",
            command=self._open,
            bg="#192b3c",
            fg=TEXT,
            activebackground="#294157",
            relief="flat",
            padx=18,
            pady=10,
        ).pack(fill="x", pady=4)

        monitor = Frame(body, bg=PANEL, padx=28, pady=25)
        monitor.pack(side=RIGHT, fill=BOTH, expand=True)
        Label(
            monitor,
            text="LIVE RESEARCH PIPELINE",
            fg=TEXT,
            bg=PANEL,
            font=("Helvetica", 12, "bold"),
        ).pack(anchor="w")
        self.pipeline = []
        for number, (title, detail) in enumerate(
            [
                ("Baseline retention", "Persistent-memory control"),
                ("Passive decay", "Time-sensitive pathway decay"),
                ("Reinforced retention", "Activation-protected pathways"),
                ("Evidence export", "CSV · JSON · publication charts"),
            ],
            1,
        ):
            row = Frame(monitor, bg="#102235", padx=15, pady=13)
            row.pack(fill="x", pady=6)
            badge = Label(
                row, text=f" {number:02d} ", fg=ACCENT, bg="#173348", font=("Helvetica", 10, "bold")
            )
            badge.pack(side=LEFT, padx=(0, 13))
            text = Frame(row, bg="#102235")
            text.pack(side=LEFT)
            Label(text, text=title, fg=TEXT, bg="#102235", font=("Helvetica", 11, "bold")).pack(
                anchor="w"
            )
            Label(text, text=detail, fg=MUTED, bg="#102235", font=("Helvetica", 9)).pack(anchor="w")
            self.pipeline.append(row)
        ttk.Progressbar(monitor, variable=self.progress, maximum=100).pack(fill="x", pady=(26, 10))
        Label(monitor, textvariable=self.status, fg=MUTED, bg=PANEL, font=("Helvetica", 10)).pack(
            anchor="w"
        )
        Label(
            monitor,
            text="Results remain exploratory until replicated on the capstone datasets.",
            fg="#f0b35a",
            bg=PANEL,
            font=("Helvetica", 9),
        ).pack(anchor="w", pady=(22, 0))

    @staticmethod
    def _field(parent: Frame, title: str, variable: IntVar, low: int, high: int) -> None:
        Label(parent, text=title, fg=MUTED, bg=PANEL, font=("Helvetica", 9)).pack(
            anchor="w", pady=(8, 3)
        )
        ttk.Spinbox(parent, from_=low, to=high, textvariable=variable, width=24).pack(fill="x")

    def _start(self) -> None:
        if any(
            thread.name == "experiment" and thread.is_alive() for thread in threading.enumerate()
        ):
            return
        self.stop_requested = False
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.latest_output = Path("results/runs") / stamp
        self.run_settings = (self.seeds.get(), self.epochs.get(), self.continual_epochs.get())
        threading.Thread(target=self._run, name="experiment", daemon=True).start()

    def _run(self) -> None:
        try:
            count, epochs, continual_epochs = self.run_settings
            seed_bank = [7, 21, 42, 84, 101, 144, 233, 377, 610, 987]
            seeds = seed_bank[:count]

            def report(message: str, progress: float) -> None:
                self.events.put(("progress", (message, progress)))

            histories, summaries = run_suite(
                self.latest_output or Path("results/runs/latest"),
                seeds,
                epochs,
                continual_epochs,
                report,
                lambda: self.stop_requested,
            )
            create_charts(histories, summaries, self.latest_output or Path("results/runs/latest"))
            self.events.put(("done", self.latest_output))
        except Exception as exc:  # GUI boundary: display unexpected failures to user
            self.events.put(("error", str(exc)))

    def _stop(self) -> None:
        self.stop_requested = True
        self.status.set("Stop requested; the current atomic run will finish safely")

    def _open(self) -> None:
        if not self.latest_output or not self.latest_output.exists():
            messagebox.showinfo("No results", "Run an experiment first.")
            return
        command = (
            ["open", str(self.latest_output)]
            if sys.platform == "darwin"
            else (
                ["explorer", str(self.latest_output)]
                if sys.platform == "win32"
                else ["xdg-open", str(self.latest_output)]
            )
        )
        subprocess.Popen(command)

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "progress":
                    message, progress = payload
                    self.status.set(str(message))
                    self.progress.set(float(progress) * 100)
                elif kind == "done":
                    self.progress.set(100)
                    self.status.set(f"Evidence bundle ready: {payload}")
                elif kind == "error":
                    self.status.set("Experiment failed")
                    messagebox.showerror("Experiment error", str(payload))
        except queue.Empty:
            pass
        self.after(100, self._poll)


def main() -> None:
    MemoryLab().mainloop()


if __name__ == "__main__":
    main()
