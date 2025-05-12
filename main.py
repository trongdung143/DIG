import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font
from PIL import Image, ImageTk
import os
import tempfile
import shutil
from tkinter import TclError
from save_face import FaceRecognitionPipeline
import tensorflow as tf
from transfer_face import TransferLearning
from predict_face import predict_face
import threading
import time


class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection")
        self.root.state("zoomed")

        self.primary_color = "#263238"
        self.secondary_color = "#37474F"
        self.accent_color = "#FF5722"
        self.accent_light = "#FFCCBC"
        self.text_color = "#ECEFF1"
        self.bg_color = "#121212"
        self.button_positive = "#4CAF50"
        self.button_neutral = "#607D8B"

        self.is_camera_on = False
        self.is_collecting = False
        self.is_processing_video = False
        self.cap = None
        self.camera_label = None
        self.init_message = None
        self.message_frame = None
        self.selected_image = None
        self.drop_info_label = None
        self.username = ""
        self.face_count = 0
        self.scanning = False

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        model1_path = "./models/face_recognition.keras"

        if os.path.exists(model1_path):
            self.model_facerecognition = tf.keras.models.load_model(model1_path)
        else:
            transfer_face = TransferLearning()
            transfer_face.transfer_learning()

        self.model_emotion = None
        dataset_path = "./data/face_data"
        self.labels = [
            name
            for name in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, name))
        ]

        self.create_header()
        self.create_main_content()
        self.setup_drag_and_drop()

    def create_header(self):
        header_frame = tk.Frame(self.root, bg=self.primary_color, height=100)
        header_frame.pack(fill=tk.X)

        logo_frame = tk.Frame(header_frame, bg=self.primary_color)
        logo_frame.pack(side=tk.LEFT, padx=20, pady=10)

        canvas = tk.Canvas(
            logo_frame, width=80, height=80, bg=self.primary_color, highlightthickness=0
        )
        canvas.pack()

        canvas.create_oval(
            5, 5, 75, 75, fill=self.accent_color, outline=self.accent_light, width=2
        )
        canvas.create_text(
            40, 40, text="FD", font=("Arial", 24, "bold"), fill=self.text_color
        )

        info_frame = tk.Frame(header_frame, bg=self.primary_color)
        info_frame.pack(side=tk.LEFT, padx=10, pady=10)

        team_name = tk.Label(
            info_frame,
            text="Nhóm 10",
            font=("Arial", 16, "bold"),
            bg=self.primary_color,
            fg=self.text_color,
        )
        team_name.pack(anchor="w")

        members_text = "Thành viên: Nguyễn Văn A - 21110037, Trần Thị B - 21110038, Lê Văn C - 21110039"
        members = tk.Label(
            info_frame,
            text=members_text,
            font=("Arial", 12),
            bg=self.primary_color,
            fg=self.text_color,
        )
        members.pack(anchor="w")

        button_frame = tk.Frame(header_frame, bg=self.primary_color)
        button_frame.pack(side=tk.RIGHT, padx=20, pady=10)

        self.select_image_btn = tk.Button(
            button_frame,
            text="Chọn Ảnh",
            command=self.select_image,
            bg=self.button_neutral,
            fg=self.text_color,
            font=("Arial", 12, "bold"),
            width=12,
            height=2,
            relief=tk.FLAT,
        )
        self.select_image_btn.pack(side=tk.LEFT, padx=10)

        self.camera_btn = tk.Button(
            button_frame,
            text="Bật Camera",
            command=self.toggle_camera,
            bg=self.accent_color,
            fg=self.text_color,
            font=("Arial", 12, "bold"),
            width=12,
            height=2,
            relief=tk.FLAT,
        )
        self.camera_btn.pack(side=tk.LEFT, padx=10)

        self.face_btn = tk.Button(
            button_frame,
            text="Thêm khuôn mặt",
            command=self.save_face,
            bg=self.accent_color,
            fg=self.text_color,
            font=("Arial", 12, "bold"),
            width=14,
            height=2,
            relief=tk.FLAT,
        )
        self.face_btn.pack(side=tk.LEFT, padx=10)

    def create_main_content(self):
        self.content_frame = tk.Frame(self.root, bg=self.bg_color)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.camera_container = tk.Frame(self.content_frame, bg="black")
        self.camera_container.pack(fill=tk.BOTH, expand=True, padx=50, pady=10)

        self.camera_label = tk.Label(self.camera_container, bg="black")
        self.camera_label.pack(fill=tk.BOTH, expand=True)

        status_frame = tk.Frame(self.root, bg=self.secondary_color, height=25)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_label = tk.Label(
            status_frame,
            text="Sẵn sàng",
            font=("Arial", 10),
            bg=self.secondary_color,
            fg=self.text_color,
        )
        self.status_label.pack(side=tk.LEFT, padx=10)

        self.root.after(100, self.show_init_message)

    def show_init_message(self):
        if hasattr(self, "message_frame") and self.message_frame is not None:
            self.message_frame.destroy()

        self.message_frame = tk.Frame(self.camera_container, bg="black")
        self.message_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.init_message = tk.Label(
            self.message_frame,
            text="Nhấn nút 'Bật Camera' hoặc 'Chọn Ảnh' để bắt đầu\nhoặc kéo thả ảnh/video vào đây",
            font=("Arial", 24, "bold"),
            bg="black",
            fg=self.text_color,
            padx=20,
            pady=20,
            justify=tk.CENTER,
        )
        self.init_message.pack()

        self.init_message.lift()

    def setup_drag_and_drop(self):
        try:
            if hasattr(self.camera_label, "drop_target_register"):
                self.camera_label.drop_target_register(DND_FILES)
                self.camera_label.dnd_bind("<<Drop>>", self.handle_drop)

                self.camera_container.drop_target_register(DND_FILES)
                self.camera_container.dnd_bind("<<Drop>>", self.handle_drop)

                self.camera_container.bind("<Enter>", self.show_drop_info)
                self.camera_container.bind("<Leave>", self.hide_drop_info)
        except (NameError, AttributeError):
            pass

    def show_drop_info(self, event):
        if (hasattr(self, "is_collecting") and self.is_collecting) or \
        (hasattr(self, "is_processing_video") and self.is_processing_video):
            return
        
        if not self.is_camera_on and self.selected_image is None:
            if self.drop_info_label is None:
                self.drop_info_label = tk.Label(
                    self.camera_container,
                    text="Thả ảnh/video vào đây",
                    font=("Arial", 14),
                    bg="black",
                    fg=self.accent_color,
                    padx=10,
                    pady=10,
                )
                self.drop_info_label.place(relx=0.5, rely=0.9, anchor=tk.CENTER)

    def hide_drop_info(self, event):
        if self.drop_info_label is not None:
            self.drop_info_label.destroy()
            self.drop_info_label = None

    def handle_drop(self, event):
        if hasattr(self, "is_collecting") and self.is_collecting:
            self.status_label.config(text="Không thể thả file trong chế độ thu thập khuôn mặt")
            return
        
        if self.is_camera_on:
            self.toggle_camera()

        file_path = self.parse_drop_event(event.data)
        
        if file_path:
            is_valid, file_type = self.is_valid_image_file(file_path)
            if is_valid:
                if hasattr(self, "message_frame") and self.message_frame is not None:
                    self.message_frame.destroy()
                
                if file_type == "image":
                    self.selected_image = Image.open(file_path)
                    self.display_selected_image()
                    self.status_label.config(text=f"Đã tải ảnh: {os.path.basename(file_path)}")
                elif file_type == "video":
                    self.process_video(file_path)
            else:
                self.status_label.config(
                    text="Không thể mở tệp. Vui lòng thả một file ảnh hoặc video hợp lệ."
                )
                
    def process_video(self, video_path):
        self.is_processing_video = True

        if self.drop_info_label is not None:
            self.drop_info_label.destroy()
            self.drop_info_label = None
        
        
        self.progress_frame = tk.Frame(self.camera_container, bg="black", padx=20, pady=20)
        self.progress_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        progress_label = tk.Label(
            self.progress_frame,
            text=f"Đang xử lý video: {os.path.basename(video_path)}",
            font=("Arial", 12),
            bg="black",
            fg=self.text_color,
            justify=tk.CENTER
        )
        progress_label.pack(pady=(0, 10))
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            variable=self.progress_var,
            length=400,
            mode="determinate"
        )
        self.progress_bar.pack(pady=5)
        
        self.progress_percent = tk.Label(
            self.progress_frame,
            text="0%",
            font=("Arial", 10),
            bg="black",
            fg=self.text_color
        )
        self.progress_percent.pack(pady=5)

        self.status_label.config(text=f"Đang xử lý video: {os.path.basename(video_path)}")
        
        processing_thread = threading.Thread(target=self.extract_frames_from_video, args=(video_path,))
        processing_thread.daemon = True
        processing_thread.start()

    def extract_frames_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.status_label.config(text=f"Không thể mở video: {os.path.basename(video_path)}")
            self.close_progress_frame()
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames <= 0:
            self.status_label.config(text="Không thể xác định số khung hình trong video")
            self.close_progress_frame()
            cap.release()
            return
        
        temp_dir = os.path.join(tempfile.gettempdir(), "face_detection_frames")
        os.makedirs(temp_dir, exist_ok=True)
        
        faces_found = 0
        frames_with_faces = []
        
        update_interval = max(1, total_frames // 100)
        
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if count % 5 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                
                if len(faces) > 0:
                    faces_found += len(faces)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        try:
                            label, confidence = predict_face(
                                frame, 0.7, self.model_facerecognition, self.labels
                            )
                            if label is not None:
                                cv2.putText(
                                    frame,
                                    f"{label} ({confidence:.2f})",
                                    (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (36, 255, 12),
                                    2,
                                )
                        except Exception as e:
                            cv2.putText(
                                frame,
                                "Unknown",
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 0, 255),
                                2,
                            )
                            print(f"Lỗi khi dự đoán khuôn mặt: {str(e)}")
                    
                    frame_path = os.path.join(temp_dir, f"frame_{count}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frames_with_faces.append(frame_path)
            
            if count % update_interval == 0:
                progress_percent = min(100, (count / total_frames) * 100)
                self.progress_var.set(progress_percent)
                
                self.root.after(0, lambda p=progress_percent: self.update_progress_text(p))
            
            count += 1
        
        cap.release()
        
        self.root.after(0, lambda: self.finish_video_processing(
            frames_with_faces, faces_found, os.path.basename(video_path), video_path))

    def update_progress_text(self, percentage):
        if hasattr(self, "progress_percent"):
            self.progress_percent.config(text=f"{percentage:.1f}%")

    def play_video(self, video_path):
        if hasattr(self, "is_playing_video") and self.is_playing_video:
            self.is_playing_video = False
            self.is_paused = True
            if hasattr(self, "play_video_btn"):
                self.play_video_btn.config(text="Tiếp tục", bg=self.button_positive)
            return
        
        if hasattr(self, "is_paused") and self.is_paused:
            self.is_playing_video = True
            self.is_paused = False
            if hasattr(self, "play_video_btn"):
                self.play_video_btn.config(text="Tạm dừng", bg=self.accent_color)
            self.display_video_frame()
            return
        
        self.is_playing_video = True
        self.is_paused = False
        
        self.select_image_btn.config(state=tk.DISABLED)
        self.camera_btn.config(state=tk.DISABLED)
        self.face_btn.config(state=tk.DISABLED)
        
        if hasattr(self, "play_video_btn"):
            self.play_video_btn.config(text="Tạm dừng", bg=self.accent_color)
        
        if hasattr(self, "message_frame") and self.message_frame is not None:
            self.message_frame.destroy()
            self.message_frame = None
        
        if self.cap is None:
            self.cap = cv2.VideoCapture(video_path)
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame = 0
            self.playback_speed = 1.0
            
            # Tạo thanh điều khiển video
            self.create_video_controls()
        
        self.status_label.config(text=f"Đang phát video: {os.path.basename(video_path)}")
        self.display_video_frame()

    def create_video_controls(self):
        if hasattr(self, "controls_frame") and self.controls_frame is not None:
            self.controls_frame.destroy()

        self.controls_frame = tk.Frame(self.camera_container, bg="black")
        self.controls_frame.place(relx=0.5, rely=0.95, anchor=tk.CENTER, width=600, height=60)

        self.seek_back_btn = tk.Button(
            self.controls_frame,
            text="⏪",
            command=lambda: self.seek_video(-30),
            bg=self.button_neutral,
                fg=self.text_color,
                font=("Arial", 12, "bold"),
            width=3,
                relief=tk.FLAT,
            )
        self.seek_back_btn.pack(side=tk.LEFT, padx=5)
        
        self.play_video_btn.destroy()
        self.play_video_btn = tk.Button(
            self.controls_frame,
            text="Tạm dừng",
            command=lambda: self.play_video(""),
            bg=self.accent_color,
            fg=self.text_color,
            font=("Arial", 12, "bold"),
            width=10,
            relief=tk.FLAT,
        )
        self.play_video_btn.pack(side=tk.LEFT, padx=5)
        
        self.seek_forward_btn = tk.Button(
            self.controls_frame,
            text="⏩",
            command=lambda: self.seek_video(30),
            bg=self.button_neutral,
            fg=self.text_color,
            font=("Arial", 12, "bold"),
            width=3,
            relief=tk.FLAT,
        )
        self.seek_forward_btn.pack(side=tk.LEFT, padx=5)
        
        speed_frame = tk.Frame(self.controls_frame, bg="black")
        speed_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Label(
            speed_frame,
            text="Tốc độ:",
            font=("Arial", 10),
            bg="black",
            fg=self.text_color
        ).pack(side=tk.LEFT)
        
        speeds = ["0.5x", "1.0x", "1.5x", "2.0x"]
        self.speed_var = tk.StringVar(value="1.0x")
        
        speed_menu = tk.OptionMenu(
            speed_frame, 
            self.speed_var, 
            *speeds,
            command=self.change_playback_speed
        )
        speed_menu.config(bg=self.button_neutral, fg=self.text_color, width=4)
        speed_menu.pack(side=tk.LEFT)
        
        self.exit_video_btn = tk.Button(
            self.controls_frame,
            text="Thoát",
            command=self.exit_video,
            bg=self.accent_color,
            fg=self.text_color,
            font=("Arial", 12, "bold"),
            width=8,
            relief=tk.FLAT,
        )
        self.exit_video_btn.pack(side=tk.RIGHT, padx=5)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_scale = ttk.Scale(
            self.controls_frame,
            variable=self.progress_var,
            from_=0,
            to=self.video_length,
            orient=tk.HORIZONTAL,
            length=200,
            command=self.seek_to_position
        )
        self.progress_scale.pack(side=tk.RIGHT, padx=10, fill=tk.X, expand=True)

    def change_playback_speed(self, speed_str):
        speed = float(speed_str.replace("x", ""))
        self.playback_speed = speed

    def seek_video(self, seconds):
        if self.cap is not None:
            frame_jump = int(seconds * self.video_fps)
            current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            new_pos = max(0, min(self.video_length - 1, current_pos + frame_jump))
            
            was_playing = self.is_playing_video and not self.is_paused
            
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            self.progress_var.set(new_pos)
            
            ret, frame = self.cap.read()
            if ret:
                self.display_single_frame(frame)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                
            if was_playing:
                self.is_playing_video = True
                self.is_paused = False
                if hasattr(self, "play_video_btn"):
                    self.play_video_btn.config(text="Tạm dừng", bg=self.accent_color)
                self.display_video_frame()

    def seek_to_position(self, value):
        if self.cap is not None:
            if not hasattr(self, "is_seeking"):
                self.is_seeking = False
                
            if not self.is_seeking:
                self.is_seeking = True
                frame_pos = int(float(value))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                
                ret, frame = self.cap.read()
                if ret:
                    self.display_single_frame(frame)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                
                self.is_seeking = False

    def display_single_frame(self, frame):
        window_width = self.camera_container.winfo_width()
        window_height = self.camera_container.winfo_height()
        
        if window_width > 1 and window_height > 1:
            height, width = frame.shape[:2]
            aspect_ratio = width / height
            
            if window_width / window_height > aspect_ratio:
                new_width = int(window_height * aspect_ratio)
                new_height = window_height
            else:
                new_width = window_width
                new_height = int(new_width / aspect_ratio)
                
            frame = cv2.resize(frame, (new_width, new_height))
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            try:
                label, confidence = predict_face(
                    frame, 0.7, self.model_facerecognition, self.labels
                )
                if label is not None:
                    cv2.putText(
                        frame,
                        f"{label} ({confidence:.2f})",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (36, 255, 12),
                        2,
                    )
            except Exception as e:
                cv2.putText(
                    frame,
                    "Unknown",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img_tk = ImageTk.PhotoImage(image=img)
        
        self.camera_label.config(image=img_tk)
        self.camera_label.image = img_tk

    def display_video_frame(self):
        if not hasattr(self, "is_playing_video") or not self.is_playing_video or self.is_paused:
            return
        
        if self.cap is None or not self.cap.isOpened():
            self.exit_video()
            return
        
        start_time = time.time()
        
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.is_playing_video = False
            self.is_paused = True
            if hasattr(self, "play_video_btn"):
                self.play_video_btn.config(text="Phát lại", bg=self.button_positive)
            return
        
        current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.progress_var.set(current_pos)
        
        if current_pos % 3 == 0:
            self.display_single_frame(frame)
        else:
            self.display_frame_without_detection(frame)
        
        processing_time = time.time() - start_time
        
        target_time = 1.0 / (self.video_fps * self.playback_speed)
        wait_time = int(max(1, (target_time - processing_time) * 1000))
        
        self.root.after(wait_time, self.display_video_frame)

    def display_frame_without_detection(self, frame):
        window_width = self.camera_container.winfo_width()
        window_height = self.camera_container.winfo_height()
        
        if window_width > 1 and window_height > 1:
            height, width = frame.shape[:2]
            aspect_ratio = width / height
            
            if window_width / window_height > aspect_ratio:
                new_width = int(window_height * aspect_ratio)
                new_height = window_height
            else:
                new_width = window_width
                new_height = int(new_width / aspect_ratio)
            
            frame = cv2.resize(frame, (new_width, new_height))
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img_tk = ImageTk.PhotoImage(image=img)
        
        self.camera_label.config(image=img_tk)
        self.camera_label.image = img_tk

    def exit_video(self):
        self.is_playing_video = False
        self.is_paused = False
        
        self.select_image_btn.config(state=tk.NORMAL)
        self.camera_btn.config(state=tk.NORMAL)
        self.face_btn.config(state=tk.NORMAL)
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        if hasattr(self, "controls_frame") and self.controls_frame is not None:
            self.controls_frame.destroy()
            self.controls_frame = None

        if hasattr(self, "result_frame") and self.result_frame is not None:
            self.result_frame.destroy()
            self.result_frame = None
        
        self.selected_image = None
        
        self.camera_label.config(image="")
        self.root.after(100, self.show_init_message)
        self.status_label.config(text="Đã thoát video")

    def finish_video_processing(self, frame_paths, faces_found, video_name, video_path):
        self.is_processing_video = False
        self.close_progress_frame()
        
        self.status_label.config(text=f"Đã tìm thấy {faces_found} khuôn mặt trong video {video_name}")
        
        if frame_paths:
            self.result_frame = tk.Frame(self.camera_container, bg="black", padx=10, pady=5)
            self.result_frame.place(relx=0.5, rely=0.85, anchor=tk.CENTER)  # Đặt ở phía dưới
            
            result_label = tk.Label(
                self.result_frame,
                text=f"Đã tìm thấy {faces_found} khuôn mặt trong {len(frame_paths)} khung hình",
                font=("Arial", 14, "bold"),
                bg="black",
                fg=self.text_color
            )
            result_label.pack()
            
            self.play_video_btn = tk.Button(
                self.camera_container,
                text="Phát Video",
                command=lambda: self.play_video(video_path),
                bg=self.button_positive,
                fg=self.text_color,
                font=("Arial", 12, "bold"),
                relief=tk.FLAT,
            )
            self.play_video_btn.place(relx=0.5, rely=0.95, anchor=tk.CENTER)
            
            self.selected_image = Image.open(frame_paths[0])
            self.display_selected_image()
            
            messagebox.showinfo(
                "Kết quả xử lý video",
                f"Đã tìm thấy {faces_found} khuôn mặt trong {len(frame_paths)} khung hình của video {video_name}.\nNhấn nút 'Phát Video' để xem video."
            )
        else:
            self.show_init_message()
            messagebox.showinfo(
                "Kết quả xử lý video",
                f"Không tìm thấy khuôn mặt nào trong video {video_name}."
            )

    def close_progress_frame(self):
        if hasattr(self, "progress_frame") and self.progress_frame is not None:
            self.progress_frame.destroy()
            self.progress_frame = None

    def parse_drop_event(self, data):
        try:
            if os.name == "nt":  # Windows
                file_path = data.strip("{}").replace("\\", "/")
            else:  # Unix systems
                if data.startswith("file:"):
                    import urllib.parse

                    file_path = urllib.parse.unquote(
                        data.strip().replace("file://", "")
                    )
                else:
                    file_path = data.strip()

            return file_path
        except Exception as e:
            self.status_label.config(text=f"Lỗi xử lý tệp: {str(e)}")
            return None

    def is_valid_image_file(self, file_path):
        valid_image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
        valid_video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv")
        
        if os.path.isfile(file_path):
            ext = os.path.splitext(file_path.lower())[1]
            if ext in valid_image_extensions:
                return True, "image"
            elif ext in valid_video_extensions:
                return True, "video"
        
        return False, None

    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("All supported files", "*.jpg *.jpeg *.png *.bmp *.gif *.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv")
            ]
        )

        if file_path:
            if self.is_camera_on:
                self.toggle_camera()

            if hasattr(self, "message_frame") and self.message_frame is not None:
                self.message_frame.destroy()

            is_valid, file_type = self.is_valid_image_file(file_path)
            if is_valid:
                if file_type == "image":
                    self.selected_image = Image.open(file_path)
                    self.display_selected_image()
                    self.status_label.config(text=f"Đã tải ảnh: {os.path.basename(file_path)}")
                elif file_type == "video":
                    self.process_video(file_path)

    def display_selected_image(self):
        if self.selected_image:
            container_width = self.camera_container.winfo_width()
            container_height = self.camera_container.winfo_height()

            if container_width <= 1:
                container_width = 800
            if container_height <= 1:
                container_height = 600

            img_width, img_height = self.selected_image.size
            aspect_ratio = img_width / img_height

            if container_width / container_height > aspect_ratio:
                new_height = container_height
                new_width = int(new_height * aspect_ratio)
            else:
                new_width = container_width
                new_height = int(new_width / aspect_ratio)

            resized_img = self.selected_image.resize(
                (new_width, new_height), Image.LANCZOS
            )

            cv_image = np.array(resized_img.convert("RGB"))
            cv_image = cv_image[:, :, ::-1].copy()

            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            for x, y, w, h in faces:
                cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                try:
                    label, confidence = predict_face(
                        cv_image, 0.7, self.model_facerecognition, self.labels
                    )
                    if label is not None:
                        cv2.putText(
                            cv_image,
                            f"{label} ({confidence:.2f})",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (36, 255, 12),
                            2,
                        )
                except Exception as e:
                    cv2.putText(
                        cv_image,
                        "Unknown",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )
                    print(f"Lỗi khi dự đoán khuôn mặt: {str(e)}")
            self.status_label.config(text=f"Phát hiện {len(faces)} khuôn mặt")
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            display_img = Image.fromarray(cv_image)
            img_tk = ImageTk.PhotoImage(image=display_img)

            self.camera_label.config(image=img_tk)
            self.camera_label.image = img_tk

            self.status_label.config(text=f"Phát hiện {len(faces)} khuôn mặt trong ảnh")

    def toggle_camera(self):
        if self.is_camera_on:
            self.is_camera_on = False
            self.camera_btn.config(text="Bật Camera", bg=self.accent_color)
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.camera_label.config(image="")
            self.root.after(100, self.show_init_message)
            self.status_label.config(text="Camera đã tắt")
        else:
            self.is_camera_on = True
            self.camera_btn.config(text="Tắt Camera", bg=self.button_positive)
            self.selected_image = None
            if hasattr(self, "message_frame") and self.message_frame is not None:
                self.message_frame.destroy()
            self.cap = cv2.VideoCapture(0)
            self.show_frame()
            self.status_label.config(text="Camera đang chạy")

    def save_face(self):
        name_dialog = tk.Toplevel(self.root)
        name_dialog.title("Nhập tên người dùng")
        name_dialog.geometry("400x150")
        name_dialog.transient(self.root)
        name_dialog.grab_set()
        name_dialog.resizable(False, False)
        
        name_dialog.update_idletasks()
        width = name_dialog.winfo_width()
        height = name_dialog.winfo_height()
        x = (name_dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (name_dialog.winfo_screenheight() // 2) - (height // 2)
        name_dialog.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        
        label_font = ("Arial", 12)
        
        tk.Label(
            name_dialog, 
            text="Nhập tên người dùng cần thu thập:", 
            font=label_font
        ).pack(pady=(20, 10))
        
        name_var = tk.StringVar()
        name_entry = tk.Entry(name_dialog, textvariable=name_var, font=label_font, width=30)
        name_entry.pack(pady=5)
        name_entry.focus_set()
        
        self.name_confirmed = False
        
        def confirm_name():
            if name_var.get().strip():
                self.username = name_var.get().strip()
                self.name_confirmed = True
                name_dialog.destroy()
                self.start_face_collection()
            else:
                messagebox.showwarning("Cảnh báo", "Vui lòng nhập tên người dùng!")
        
        tk.Button(
            name_dialog, 
            text="Xác nhận", 
            command=confirm_name,
            bg=self.button_positive, 
            fg=self.text_color,
            font=("Arial", 11),
            width=10
        ).pack(pady=15)
        
        def on_close():
            self.name_confirmed = False
            name_dialog.destroy()
        
        name_dialog.protocol("WM_DELETE_WINDOW", on_close)
        
        self.root.wait_window(name_dialog)

    def start_face_collection(self):
        if self.is_camera_on:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.camera_label.config(image="")
        
        self.is_collecting = True
        self.face_count = 0
        self.max_faces = 300
        self.scanning = False
        
        base_dir = "./data/face_data"
        self.save_dir = os.path.join(base_dir, self.username)
        os.makedirs(self.save_dir, exist_ok=True)
        
        if hasattr(self, "message_frame") and self.message_frame is not None:
            self.message_frame.destroy()
            self.message_frame = None
        
        if self.drop_info_label is not None:
            self.drop_info_label.destroy()
            self.drop_info_label = None
        
        self.select_image_btn.config(state=tk.DISABLED)
        self.camera_btn.config(state=tk.DISABLED)
        self.face_btn.config(state=tk.DISABLED)
        
        self.cancel_collect_btn = tk.Button(
            self.camera_container,
            text="Dừng thu thập",
            command=self.cancel_collection,
            bg=self.accent_color,
            fg=self.text_color,
            font=("Arial", 12, "bold"),
            relief=tk.FLAT,
        )
        self.cancel_collect_btn.place(relx=0.5, rely=0.95, anchor=tk.CENTER)
        
        self.status_label.config(text=f"Đang thu thập khuôn mặt cho: {self.username}")
        
        self.cap = cv2.VideoCapture(0)
        self.collect_faces()
        
    def collect_faces(self):
        if not self.is_collecting:
            return
        
        ret, frame = self.cap.read()
        if ret:
            # Điều chỉnh kích thước frame theo container
            window_width = self.camera_container.winfo_width()
            window_height = self.camera_container.winfo_height()
            
            if window_width > 1 and window_height > 1:
                height, width = frame.shape[:2]
                aspect_ratio = width / height
                
                if window_width / window_height > aspect_ratio:
                    new_width = int(window_height * aspect_ratio)
                    new_height = window_height
                else:
                    new_width = window_width
                    new_height = int(new_width / aspect_ratio)
                    
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Phát hiện khuôn mặt
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            annotated = frame.copy()
            
            if self.scanning and len(faces) > 1:
                self.status_label.config(text="Quá nhiều khuôn mặt! Vui lòng chỉ hiển thị một khuôn mặt")
            
            if not self.scanning and len(faces) == 1:
                self.scanning = True
                self.status_label.config(text=f"Bắt đầu quét cho: {self.username}")
            
            for x, y, w, h in faces:
                if len(faces) == 1:
                    face_crop = frame[y:y+h, x:x+w]
                    filename = os.path.join(self.save_dir, f"face_{self.face_count + 1}.jpg")
                    cv2.imwrite(filename, face_crop)
                    self.face_count += 1
                    
                cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            cv_font = cv2.FONT_HERSHEY_SIMPLEX
            
            cv2.putText(
                annotated,
                f"Da thu thap: {self.face_count}/{self.max_faces}",
                (10, 30),
                cv_font,
                0.8,
                (0, 255, 0),
                2
            )
            
            self.status_label.config(text=f"Đã thu thập: {self.face_count}/{self.max_faces} cho {self.username}")
            
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(annotated)
            img_tk = ImageTk.PhotoImage(image=img)
            
            self.camera_label.config(image=img_tk)
            self.camera_label.image = img_tk
            
            if self.face_count >= self.max_faces:
                self.status_label.config(text=f"Đã thu thập đủ {self.max_faces} ảnh. Hoàn tất!")
                self.finish_collection()
                return
        
        self.root.after(10, self.collect_faces)

    def cancel_collection(self):
        self.is_collecting = False
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        if hasattr(self, "cancel_collect_btn") and self.cancel_collect_btn is not None:
            self.cancel_collect_btn.destroy()
            self.cancel_collect_btn = None
        
        self.select_image_btn.config(state=tk.NORMAL)
        self.camera_btn.config(state=tk.NORMAL)
        self.face_btn.config(state=tk.NORMAL)
        
        self.camera_label.config(image="")
        self.root.after(100, self.show_init_message)
        self.status_label.config(text="Đã dừng thu thập khuôn mặt")

    def finish_collection(self):
        self.is_collecting = False
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        if hasattr(self, "cancel_collect_btn") and self.cancel_collect_btn is not None:
            self.cancel_collect_btn.destroy()
            self.cancel_collect_btn = None
        
        self.select_image_btn.config(state=tk.NORMAL)
        self.camera_btn.config(state=tk.NORMAL)
        self.face_btn.config(state=tk.NORMAL)
        
        messagebox.showinfo(
            "Hoàn tất",
            f"Đã thu thập thành công {self.face_count} ảnh cho {self.username}.\n"
            "Dữ liệu đã được lưu vào thư mục dữ liệu."
        )

        self.camera_label.config(image="")
        self.root.after(100, self.show_init_message)
        self.status_label.config(text=f"Thu thập khuôn mặt hoàn tất cho {self.username}")

    def show_frame(self):
        if not self.is_camera_on:
            return

        ret, frame = self.cap.read()
        if ret:
            window_width = self.camera_container.winfo_width()
            window_height = self.camera_container.winfo_height()

            if window_width > 1 and window_height > 1:
                height, width = frame.shape[:2]
                aspect_ratio = width / height

                if window_width / window_height > aspect_ratio:
                    new_width = int(window_height * aspect_ratio)
                    new_height = window_height
                else:
                    new_width = window_width
                    new_height = int(new_width / aspect_ratio)

                frame = cv2.resize(frame, (new_width, new_height))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            try:
                label, confidence = predict_face(
                    frame, 0.7, self.model_facerecognition, self.labels
                )
                if label is not None:
                    cv2.putText(
                        frame,
                        f"{label} ({confidence:.2f})",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (36, 255, 12),
                        2,
                    )
            except Exception as e:
                cv2.putText(
                    frame,
                    "Unknown",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                print(f"Lỗi khi dự đoán khuôn mặt: {str(e)}")

            self.status_label.config(text=f"Phát hiện {len(faces)} khuôn mặt")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img_tk = ImageTk.PhotoImage(image=img)

            self.camera_label.config(image=img_tk)
            self.camera_label.image = img_tk

        self.root.after(10, self.show_frame)


def main():
    try:
        import tkinterdnd2

        global DND_FILES
        from tkinterdnd2 import DND_FILES, TkinterDnD

        root = TkinterDnD.Tk()
        app = CameraApp(root)
        root.mainloop()
    except ImportError:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(
            "Thư viện thiếu",
            "Cần cài đặt thư viện tkinterdnd2 để hỗ trợ kéo thả.\n"
            "Vui lòng cài đặt bằng lệnh: pip install tkinterdnd2\n\n"
            "Ứng dụng sẽ chạy mà không có tính năng kéo thả.",
        )

        root.destroy()
        root = tk.Tk()
        app = CameraApp(root)
        root.mainloop()


if __name__ == "__main__":
    main()
