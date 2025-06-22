import tkinter as tk
from PIL import Image, ImageTk
import cv2
import time
from typing import Tuple

import torch

from rtdetr.model import RTDETR


class Dashboard:
    """Simple local dashboard showing webcam feed and placeholders."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Detection Dashboard")

        # top-left: current time
        self.time_label = tk.Label(self.root, text="")
        self.time_label.grid(row=0, column=0, sticky="w", padx=10, pady=5)

        # top-right: current sample information
        self.sample_label = tk.Label(self.root, text="Sample: -")
        self.sample_label.grid(row=0, column=1, sticky="e", padx=10, pady=5)

        # center: captured image
        self.image_label = tk.Label(self.root)
        self.image_label.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

        # below image: detection result
        self.result_label = tk.Label(self.root, text="Detection: -")
        self.result_label.grid(row=2, column=0, columnspan=2, pady=5)

        # below detection result: detection history
        self.history_text = tk.Text(self.root, height=8, width=60)
        self.history_text.grid(row=3, column=0, columnspan=2, padx=10, pady=5)

        self.cap = cv2.VideoCapture(0)

        # simple RTDETR model with random weights
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RTDETR(num_classes=80).to(self.device)
        self.model.eval()

    def update_time(self):
        self.time_label.config(text=time.strftime("%Y-%m-%d %H:%M:%S"))
        self.root.after(1000, self.update_time)

    def detect(self, frame) -> Tuple[str, list]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)
        tensor = tensor / 255.0
        tensor = tensor.to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
        logits = outputs["pred_logits"][0]
        boxes = outputs["pred_boxes"][0]
        cls_id = int(logits.argmax(-1))
        score = float(logits.softmax(-1)[cls_id])
        return f"class {cls_id} ({score:.2f})", boxes.tolist()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            result, box = self.detect(frame)
            h, w = frame.shape[:2]
            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            img = img.resize((640, 480))
            self.photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.photo)

            self.sample_label.config(
                text=f"Sample {frame.shape[1]}x{frame.shape[0]}")
            self.result_label.config(text=f"Detection: {result}")
            self.history_text.insert(tk.END, result + "\n")
            self.history_text.see(tk.END)
        self.root.after(50, self.update_frame)

    def run(self):
        self.update_time()
        self.update_frame()
        self.root.mainloop()


if __name__ == "__main__":
    Dashboard().run()

