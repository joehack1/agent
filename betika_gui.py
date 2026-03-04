from __future__ import annotations

import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText


APP_TITLE = "Betika Bot GUI"


class BetikaGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("760x560")
        self.root.minsize(680, 500)

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        self.reader_thread: threading.Thread | None = None

        self.script_path = Path(__file__).resolve().with_name("betika.py")

        self._build_ui()
        self._schedule_log_poll()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        controls = ttk.LabelFrame(frame, text="Run Settings", padding=10)
        controls.pack(fill=tk.X)

        self.max_odds_var = tk.StringVar(value="1.35")
        self.count_var = tk.StringVar(value="39")
        self.stake_var = tk.StringVar(value="2")
        self.min_odds_var = tk.StringVar(value="1.01")
        self.execute_var = tk.BooleanVar(value=False)
        self.keep_open_var = tk.BooleanVar(value=True)

        ttk.Label(controls, text="Odds Limit (max):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(controls, textvariable=self.max_odds_var, width=12).grid(
            row=0, column=1, sticky=tk.W, padx=5, pady=5
        )

        ttk.Label(controls, text="No. of Selections:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(controls, textvariable=self.count_var, width=12).grid(
            row=0, column=3, sticky=tk.W, padx=5, pady=5
        )

        ttk.Label(controls, text="Stake (KES):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(controls, textvariable=self.stake_var, width=12).grid(
            row=1, column=1, sticky=tk.W, padx=5, pady=5
        )

        ttk.Label(controls, text="Min Odds:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(controls, textvariable=self.min_odds_var, width=12).grid(
            row=1, column=3, sticky=tk.W, padx=5, pady=5
        )

        ttk.Checkbutton(
            controls,
            text="Live Bet (--execute)",
            variable=self.execute_var,
        ).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)

        ttk.Checkbutton(
            controls,
            text="Keep Browser Open",
            variable=self.keep_open_var,
        ).grid(row=2, column=2, columnspan=2, sticky=tk.W, padx=5, pady=5)

        for col in range(4):
            controls.grid_columnconfigure(col, weight=1)

        buttons = ttk.Frame(frame)
        buttons.pack(fill=tk.X, pady=(8, 8))

        self.run_btn = ttk.Button(buttons, text="Start Bot", command=self.start_run)
        self.run_btn.pack(side=tk.LEFT)

        self.stop_btn = ttk.Button(buttons, text="Stop", command=self.stop_run, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=8)

        clear_btn = ttk.Button(buttons, text="Clear Log", command=self.clear_log)
        clear_btn.pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(buttons, textvariable=self.status_var).pack(side=tk.RIGHT)

        log_frame = ttk.LabelFrame(frame, text="Output", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = ScrolledText(log_frame, wrap=tk.WORD, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.configure(state=tk.DISABLED)

    def start_run(self) -> None:
        if self.process is not None:
            self._append_log("A run is already active.\n")
            return

        try:
            max_odds = float(self.max_odds_var.get().strip())
            min_odds = float(self.min_odds_var.get().strip())
            count = int(self.count_var.get().strip())
            stake = float(self.stake_var.get().strip())
        except ValueError:
            self._append_log("Invalid numeric input. Check odds, count, and stake.\n")
            return

        if count <= 0:
            self._append_log("No. of selections must be > 0.\n")
            return
        if stake <= 0:
            self._append_log("Stake must be > 0.\n")
            return
        if min_odds <= 0 or max_odds <= 0:
            self._append_log("Odds values must be > 0.\n")
            return
        if min_odds > max_odds:
            self._append_log("Min odds cannot be greater than odds limit.\n")
            return

        cmd = [
            sys.executable,
            str(self.script_path),
            "--max-odds",
            f"{max_odds}",
            "--min-odds",
            f"{min_odds}",
            "--count",
            str(count),
            "--stake",
            f"{stake}",
        ]

        if self.execute_var.get():
            cmd.append("--execute")
        if self.keep_open_var.get():
            cmd.append("--keep-open")

        self._append_log(f"Starting: {' '.join(cmd)}\n")

        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=str(self.script_path.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            self._append_log(f"Failed to start process: {exc}\n")
            self.process = None
            return

        self.run_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.status_var.set("Running...")

        self.reader_thread = threading.Thread(target=self._read_output, daemon=True)
        self.reader_thread.start()

    def stop_run(self) -> None:
        if self.process is None:
            return
        self._append_log("Stopping process...\n")
        self.process.terminate()

    def clear_log(self) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _read_output(self) -> None:
        assert self.process is not None
        proc = self.process

        if proc.stdout is not None:
            for line in proc.stdout:
                self.log_queue.put(line)

        return_code = proc.wait()
        self.log_queue.put(f"\nProcess exited with code {return_code}\n")
        self.log_queue.put("__RUN_FINISHED__")

    def _schedule_log_poll(self) -> None:
        self.root.after(120, self._poll_log_queue)

    def _poll_log_queue(self) -> None:
        try:
            while True:
                msg = self.log_queue.get_nowait()
                if msg == "__RUN_FINISHED__":
                    self.process = None
                    self.reader_thread = None
                    self.run_btn.configure(state=tk.NORMAL)
                    self.stop_btn.configure(state=tk.DISABLED)
                    self.status_var.set("Idle")
                else:
                    self._append_log(msg)
        except queue.Empty:
            pass
        finally:
            self._schedule_log_poll()

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)


def main() -> int:
    root = tk.Tk()
    app = BetikaGui(root)
    def on_close() -> None:
        app.stop_run()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
