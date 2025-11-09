import cv2
from deepface import DeepFace
import numpy as np
import os
import customtkinter as ctk
from PIL import Image, ImageTk
import threading
from datetime import datetime

class FaceRecognitionApp:
    def _init_(self, root):
        self.root = root
        self.root.title("AI Face Recognition System")
        self.root.geometry("950x750")
        ctk.set_appearance_mode("dark")

        self.cap = None
        self.running = False

        # --- UI ---
        self.title_label = ctk.CTkLabel(root, text="🧠 Face Detection + Recognition (DeepFace)", font=("Arial", 28, "bold"))
        self.title_label.pack(pady=15)

        self.video_label = ctk.CTkLabel(root, text="")
        self.video_label.pack(pady=10)

        self.status_label = ctk.CTkLabel(root, text="Status: Idle", font=("Arial", 18))
        self.status_label.pack(pady=10)

        # --- Buttons ---
        self.start_btn = ctk.CTkButton(root, text="▶ Start Recognition", command=self.start_recognition)
        self.start_btn.pack(side="left", padx=25, pady=20)

        self.stop_btn = ctk.CTkButton(root, text="⏸ Stop", command=self.stop_recognition)
        self.stop_btn.pack(side="left", padx=25, pady=20)

        self.capture_btn = ctk.CTkButton(root, text="📸 Capture Snapshot", command=self.capture_snapshot)
        self.capture_btn.pack(side="left", padx=25, pady=20)

        self.register_btn = ctk.CTkButton(root, text="🧍 Register New Face", command=self.register_new_face)
        self.register_btn.pack(side="left", padx=25, pady=20)

        # --- Load known faces ---
        self.known_faces_dir = "known_faces"
        os.makedirs(self.known_faces_dir, exist_ok=True)
        self.known_faces = [f for f in os.listdir(self.known_faces_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.status_label.configure(text=f"✅ Loaded {len(self.known_faces)} known faces")

    def start_recognition(self):
        if not self.running:
            self.running = True
            self.cap = cv2.VideoCapture(0)
            threading.Thread(target=self.recognition_loop, daemon=True).start()

    def stop_recognition(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.status_label.configure(text="⏹ Recognition stopped")

    def recognition_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            try:
                # DeepFace face detection and analysis
                detections = DeepFace.extract_faces(frame, detector_backend='opencv', enforce_detection=False)
            except Exception as e:
                print(f"Error: {e}")
                detections = []

            for det in detections:
                region = det['facial_area']
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                face = frame[y:y + h, x:x + w]
                name = "Unknown"

                # Try to match with known faces
                for known in self.known_faces:
                    known_path = os.path.join(self.known_faces_dir, known)
                    try:
                        result = DeepFace.verify(face, known_path, model_name='Facenet', enforce_detection=False)
                        if result['verified']:
                            name = os.path.splitext(known)[0]
                            break
                    except:
                        continue

                # Draw bounding box
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.rectangle(frame, (x, y + h - 35), (x + w, y + h), color, cv2.FILLED)
                cv2.putText(frame, name, (x + 6, y + h - 6),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            self.status_label.configure(text=f"Faces Detected: {len(detections)}")

        if self.cap:
            self.cap.release()

    def capture_snapshot(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                filename = f"snapshot_{datetime.now().strftime('%H-%M-%S')}.jpg"
                cv2.imwrite(filename, frame)
                self.status_label.configure(text=f"📷 Snapshot saved: {filename}")

    def register_new_face(self):
        """Capture and register a new face directly from webcam."""
        name_window = ctk.CTkInputDialog(text="Enter name for the new face:", title="Register New Face")
        name = name_window.get_input()
        if not name:
            self.status_label.configure(text="❌ Registration canceled (no name).")
            return

        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            self.status_label.configure(text="⚠ Failed to capture image.")
            return

        detections = DeepFace.extract_faces(frame, detector_backend='opencv', enforce_detection=False)
        if not detections:
            self.status_label.configure(text="❌ No face detected. Try again.")
            return

        face_data = detections[0]['face']
        face_image = np.array(face_data * 255, dtype=np.uint8)
        filename = os.path.join(self.known_faces_dir, f"{name}.jpg")
        cv2.imwrite(filename, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))

        self.known_faces.append(f"{name}.jpg")
        self.status_label.configure(text=f"✅ New face '{name}' registered.")

# --- Run App ---
if __name__ == "_main_":
    root = ctk.CTk()
    app = FaceRecognitionApp(root)
    root.mainloop()