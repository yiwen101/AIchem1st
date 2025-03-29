import platform
import subprocess
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import pandas as pd


class VideoLabellerApp:
    INPUT_FILE_PATH = "mcq_part_2.parquet"
    OUTPUT_FILE_PATH = "labelled_mcq_part_2.parquet"
    df: pd.DataFrame

    def __init__(self, root):
        self.root = root
        self.root.title("MCQ Video Labeller")
        self.root.geometry("600x600")

        # Data handling
        self.current_index = 0

        self.setup_ui()
        self.load_data()

    def setup_ui(self):
        # Question frame
        question_frame = ttk.LabelFrame(self.root, text="Question", padding=10)
        question_frame.pack(fill="x", padx=10, pady=5)

        self.qid_label = ttk.Label(question_frame, text="")
        self.qid_label.pack(anchor="w")

        self.question_text = tk.Text(question_frame, height=4, wrap="word")
        self.question_text.pack(fill="x")

        # Options frame
        options_frame = ttk.LabelFrame(self.root, text="Options", padding=10)
        options_frame.pack(fill="x", padx=10, pady=5)

        self.options_text = tk.Text(options_frame, height=6, wrap="word")
        self.options_text.pack(fill="x")

        # Controls frame
        controls_frame = ttk.LabelFrame(self.root, text="Controls", padding=10)
        controls_frame.pack(fill="x", padx=10, pady=5)

        # Video button
        ttk.Button(controls_frame, text="Play Video", command=self.play_video).pack(
            fill="x", pady=5
        )

        # Answer buttons
        answers_frame = ttk.Frame(controls_frame)
        answers_frame.pack(fill="x", pady=5)

        answers = ["A", "B", "C", "D", "E"]
        for answer in answers:
            btn = ttk.Button(
                answers_frame,
                text=answer,
                command=lambda a=answer: self.submit_answer(a),
                width=5,
            )
            btn.pack(side="left", padx=2, expand=True)

        # Navigation buttons
        nav_frame = ttk.Frame(controls_frame)
        nav_frame.pack(fill="x", pady=5)

        ttk.Button(nav_frame, text="Previous", command=self.prev_question).pack(
            side="left", padx=5, expand=True
        )
        ttk.Button(nav_frame, text="Next", command=self.next_question).pack(
            side="left", padx=5, expand=True
        )
        ttk.Button(nav_frame, text="Save", command=self.save_data).pack(
            side="left", padx=5, expand=True
        )

        # Quit button
        ttk.Button(
            controls_frame, text="Quit", command=self.on_closing, style="Accent.TButton"
        ).pack(fill="x", pady=5)

        # Progress label
        self.progress_label = ttk.Label(self.root, text="")
        self.progress_label.pack(pady=5)

    def load_data(self):
        self.df = pd.read_parquet(self.INPUT_FILE_PATH, engine="pyarrow")
        self.show_current_question()

    def show_current_question(self):
        row = self.df.iloc[self.current_index]

        # Update question and options
        self.qid_label.config(text=f"Question ID: {row['qid']}")

        question_lines = row["question"].split("\n")
        question = question_lines[0]
        options = question_lines[1:]

        self.question_text.delete("1.0", tk.END)
        self.question_text.insert("1.0", question)

        self.options_text.delete("1.0", tk.END)
        options_text = "\n".join(options)
        options_text += "\nE. None of the above"
        self.options_text.insert("1.0", options_text)

        # Update progress
        total = len(self.df)
        labeled = len(self.df[self.df["answer"] != ""])
        self.progress_label.config(
            text=f"Progress: {labeled}/{total} questions labeled"
        )

    def play_video(self):
        row = self.df.iloc[self.current_index]
        vid = row["youtube_url"].split("/")[-1]
        video_path = str(Path("videos") / f"{vid}.mp4")

        # Open video with system default player
        if platform.system() == "Darwin":  # macOS
            subprocess.run(["open", video_path])
        elif platform.system() == "Windows":
            subprocess.run(["start", video_path], shell=True)
        else:  # Linux
            subprocess.run(["xdg-open", video_path])

    def submit_answer(self, answer):
        self.df.loc[self.current_index, "answer"] = answer
        self.next_question()

    def next_question(self):
        if self.current_index < len(self.df) - 1:
            self.current_index += 1
            self.show_current_question()

    def prev_question(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_question()

    def save_data(self):
        self.df.to_parquet(self.OUTPUT_FILE_PATH, engine="pyarrow")

    def on_closing(self):
        self.root.destroy()


def main():
    root = tk.Tk()

    # Create a style for the quit button
    style = ttk.Style()
    style.configure("Accent.TButton", foreground="red")

    app = VideoLabellerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
