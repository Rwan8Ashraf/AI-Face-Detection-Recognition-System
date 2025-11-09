import cv2
import mediapipe as mp
import customtkinter as ctk
from PIL import Image, ImageTk
import threading

# --- Face Detection Setup ---
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# --- App Class ---
class FaceDetectionApp:
    def _init_(self, root):
        self.root = root
        self.root.title("AI Face Detection App")
        self.root.geometry("900x700")
        ctk.set_appearance_mode("dark")

        # --- Camera Setup ---
        self.cap = cv2.VideoCapture(0)
        self.running = False

        # --- Widgets ---
        self.title_label = ctk.CTkLabel(root, text="👁 Face Detection System", font=("Arial", 28, "bold"))
        self.title_label.pack(pady=10)

        self.video_label = ctk.CTkLabel(root, text="")
        self.video_label.pack(pady=10)

        self.face_count_label = ctk.CTkLabel(root, text="Faces Detected: 0", font=("Arial", 18))
        self.face_count_label.pack(pady=10)

        self.start_btn = ctk.CTkButton(root, text="▶ Start Detection", command=self.start_detection)
        self.start_btn.pack(side="left", padx=30, pady=20)

        self.stop_btn = ctk.CTkButton(root, text="⏸ Stop Detection", command=self.stop_detection)
        self.stop_btn.pack(side="left", padx=30, pady=20)

        self.capture_btn = ctk.CTkButton(root, text="📸 Capture Snapshot", command=self.capture_snapshot)
        self.capture_btn.pack(side="left", padx=30, pady=20)

    def start_detection(self):
        if not self.running:
            self.running = True
            threading.Thread(target=self.detect_faces, daemon=True).start()

    def stop_detection(self):
        self.running = False

    def detect_faces(self):
        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(frame_rgb)

                face_count = 0
                if results.detections:
                    for detection in results.detections:
                        mp_drawing.draw_detection(frame, detection)
                    face_count = len(results.detections)

                self.face_count_label.configure(text=f"Faces Detected: {face_count}")

                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

        self.cap.release()

    def capture_snapshot(self):
        ret, frame = self.cap.read()
        if ret:
            filename = "snapshot.jpg"
            cv2.imwrite(filename, frame)
            ctk.CTkLabel(self.root, text=f"Snapshot saved: {filename}", font=("Arial", 14)).pack()

# --- Run App ---
if __name__ == "_main_":
    root = ctk.CTk()
    app = FaceDetectionApp(root)
    root.mainloop()