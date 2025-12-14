import tkinter as tk
from tkinter import simpledialog, messagebox
import cv2
from PIL import Image, ImageTk
from face_lib import FaceSystem
import threading

class FaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition App (MobileNetV2)")
        self.root.geometry("1000x700")
        
        # Initialize Backend
        self.fs = FaceSystem()
        
        # State variables
        self.is_recognizing = False
        self.is_collecting = False
        self.collect_name = ""
        self.collect_count = 0
        self.collect_limit = 50  # Number of images to collect
        
        # UI Layout
        self.setup_ui()
        
        # Camera
        self.cap = cv2.VideoCapture(0)
        
        # Start Video Loop
        self.update_video()

    def setup_ui(self):
        # Top Frame for Video
        self.video_frame = tk.Frame(self.root, bg="black")
        self.video_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack(expand=True)
        
        # Bottom Frame for Controls
        self.control_frame = tk.Frame(self.root, height=100)
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        self.btn_register = tk.Button(self.control_frame, text="Register New User", command=self.start_registration, height=2, width=20)
        self.btn_register.pack(side=tk.LEFT, padx=10)
        
        self.btn_train = tk.Button(self.control_frame, text="Train Model", command=self.start_training, height=2, width=20)
        self.btn_train.pack(side=tk.LEFT, padx=10)
        
        self.btn_recognize = tk.Button(self.control_frame, text="Start Recognition", command=self.toggle_recognition, height=2, width=20, bg="#dddddd")
        self.btn_recognize.pack(side=tk.LEFT, padx=10)
        
        self.status_label = tk.Label(self.control_frame, text="Status: Ready", font=("Arial", 12))
        self.status_label.pack(side=tk.RIGHT, padx=10)

    def start_registration(self):
        name = simpledialog.askstring("Input", "Enter Name for the user:")
        if name:
            self.collect_name = name.strip()
            self.collect_count = 0
            self.is_collecting = True
            self.is_recognizing = False
            self.status_label.config(text=f"Status: Collecting data for {name}...")
            
    def start_training(self):
        self.status_label.config(text="Status: Training... Please wait.")
        self.root.update()
        
        # Run training in separate thread to not freeze UI
        def train_task():
            res = self.fs.train(epochs=5) # 5 epochs for quick demo
            self.root.after(0, lambda: self.status_label.config(text=f"Status: {res}"))
            self.fs.load_resources() # Reload new model
            
        threading.Thread(target=train_task, daemon=True).start()

    def toggle_recognition(self):
        if self.is_recognizing:
            self.is_recognizing = False
            self.btn_recognize.config(text="Start Recognition", bg="#dddddd")
            self.status_label.config(text="Status: Ready")
        else:
            if self.fs.model is None:
                messagebox.showerror("Error", "No model trained yet! Register users and Train first.")
                return
            self.is_recognizing = True
            self.btn_recognize.config(text="Stop Recognition", bg="#90ee90")
            self.status_label.config(text="Status: Recognizing...")

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            # Face Detection
            faces = self.fs.detect_faces(frame)
            
            # Logic based on state
            if self.is_collecting:
                for (x, y, w, h) in faces:
                    # Draw rectangle green
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    if self.collect_count < self.collect_limit:
                        self.fs.save_face(frame, (x,y,w,h), self.collect_name)
                        self.collect_count += 1
                        self.status_label.config(text=f"Collected {self.collect_count}/{self.collect_limit}")
                    else:
                        self.is_collecting = False
                        self.status_label.config(text=f"Status: Collection for {self.collect_name} Done!")
                        messagebox.showinfo("Info", "Data Collection Complete!")
                    # Only collect one face per frame to avoid duplicates/confusion
                    break 

            elif self.is_recognizing:
                for (x, y, w, h) in faces:
                    face_roi = frame[y:y+h, x:x+w]
                    name, conf = self.fs.predict(face_roi)
                    
                    color = (0, 255, 0)
                    if name == "Unknown" or conf < 0.5:
                        name = "Unknown"
                        color = (0, 0, 255)
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Create label with name and confidence
                    label = f"{name} ({conf:.1%})"
                    
                    # Get text size
                    (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    
                    # Draw filled rectangle for text background
                    cv2.rectangle(frame, (x, y - 30), (x + w_text, y), color, -1)
                    
                    # Draw text in white on top of the filled rectangle
                    cv2.putText(frame, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            else:
                # Just draw boxes
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Convert to Tkinter Format
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        
        self.root.after(10, self.update_video)

    def on_closing(self):
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
