import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
import mediapipe as mp
import pickle
import os

# --- Load the embedding model with a custom layer ---
class L2NormalizeLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

with custom_object_scope({"L2NormalizeLayer": L2NormalizeLayer}):
    embedding_model = load_model("fixed_model.h5")

# --- Initialize MediaPipe Face Detection ---
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# --- Global Variables ---
detection_enabled = False       # flag for simple face detection mode
live_view_enabled = False       # flag for live recognition mode
latest_bbox = None              # store latest detected face bbox (x, y, w, h)
latest_frame = None             # store latest frame
embeddings_file = "embeddings.pkl"

# --- Load Saved Embeddings (if any) ---
if os.path.exists(embeddings_file):
    with open(embeddings_file, "rb") as f:
        embeddings_dict = pickle.load(f)
else:
    embeddings_dict = {}

def save_embeddings():
    with open(embeddings_file, "wb") as f:
        pickle.dump(embeddings_dict, f)

# --- Open Video Capture ---
cap = cv2.VideoCapture(1)  # Change index if needed (0 for default webcam)

# --- Create Tkinter UI ---
root = tk.Tk()
root.title("Facial Recognition UI")

# Label to show video feed
video_label = tk.Label(root)
video_label.pack()

# Entry widget for user's name
name_frame = tk.Frame(root)
name_frame.pack(pady=5)
tk.Label(name_frame, text="Enter your name:").pack(side=tk.LEFT)
name_entry = tk.Entry(name_frame)
name_entry.pack(side=tk.LEFT)

# Status label to show messages
status_label = tk.Label(root, text="Status: Ready")
status_label.pack(pady=5)

# --- Function to Update Video Frame ---
def update_frame():
    global latest_frame, latest_bbox
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    # Save a copy for later processing
    latest_frame = frame.copy()

    # --- Live View Mode: Detection + Recognition ---
    if live_view_enabled:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb_frame)
        if results.detections:
            # Process every detected face
            for i, detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)
                # Save first detection for capture/verification functions
                if i == 0:
                    latest_bbox = (x, y, width, height)
                # Crop face region (with basic boundary checks)
                face_img = frame[y:y+height, x:x+width]
                if face_img.size == 0:
                    continue
                face_resized = cv2.resize(face_img, (128, 128))
                face_array = face_resized.astype("float32") / 255.0
                face_array = np.expand_dims(face_array, axis=0)
                new_embedding = embedding_model.predict(face_array)
                
                # Default to "Unknown" unless a match is found
                identity = "Unknown"
                min_distance = float('inf')
                threshold = 0.35  # Distance threshold for recognition
                
                # Compare against all stored embeddings
                for name, stored_embedding_list in embeddings_dict.items():
                    stored_embedding = np.array(stored_embedding_list)
                    distance = np.linalg.norm(new_embedding - stored_embedding)
                    if distance < min_distance:
                        min_distance = distance
                        identity = name
                if min_distance >= threshold:
                    identity = "Unknown"
                
                # Draw red bounding box and put the recognized name
                cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 0, 255), 2)
                cv2.putText(frame, identity, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            latest_bbox = None

    # --- Simple Face Detection Mode ---
    elif detection_enabled:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb_frame)
        if results.detections:
            detection = results.detections[0]  # use the first detected face
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x = int(bboxC.xmin * w)
            y = int(bboxC.ymin * h)
            width = int(bboxC.width * w)
            height = int(bboxC.height * h)
            latest_bbox = (x, y, width, height)
            cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
        else:
            latest_bbox = None

    # --- Display the frame in Tkinter ---
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    root.after(10, update_frame)

# --- Button Callback Functions ---
def toggle_detection():
    global detection_enabled, live_view_enabled
    # Disable live view if active
    if live_view_enabled:
        live_view_enabled = False
    detection_enabled = not detection_enabled
    if detection_enabled:
        status_label.config(text="Status: Face Detection Enabled")
    else:
        status_label.config(text="Status: Face Detection Disabled")

def toggle_live_view():
    global live_view_enabled, detection_enabled
    # Disable simple detection if active
    if detection_enabled:
        detection_enabled = False
    live_view_enabled = not live_view_enabled
    if live_view_enabled:
        status_label.config(text="Status: Live View Enabled")
    else:
        status_label.config(text="Status: Live View Disabled")

def capture_face():
    global latest_frame, latest_bbox, embeddings_dict
    if latest_frame is None:
        messagebox.showerror("Error", "No frame available!")
        return
    if latest_bbox is None:
        messagebox.showerror("Error", "No face detected!")
        return

    x, y, w_box, h_box = latest_bbox
    face_img = latest_frame[y:y+h_box, x:x+w_box]
    if face_img.size == 0:
        messagebox.showerror("Error", "Failed to crop face!")
        return

    face_resized = cv2.resize(face_img, (128, 128))
    face_array = face_resized.astype("float32") / 255.0
    face_array = np.expand_dims(face_array, axis=0)
    embedding = embedding_model.predict(face_array)
    
    user_name = name_entry.get().strip()
    if user_name == "":
        messagebox.showerror("Error", "Please enter your name!")
        return

    embeddings_dict[user_name] = embedding.tolist()
    save_embeddings()
    status_label.config(text=f"Status: Embedding saved for {user_name}")

def verify_face():
    global latest_frame, latest_bbox, embeddings_dict
    user_name = name_entry.get().strip()
    if user_name == "":
        messagebox.showerror("Error", "Please enter your name for verification!")
        return
    if user_name not in embeddings_dict:
        messagebox.showerror("Error", "No stored embedding for this user!")
        return

    stored_embedding = np.array(embeddings_dict[user_name])
    if latest_frame is None:
        messagebox.showerror("Error", "No frame available!")
        return
    if latest_bbox is None:
        messagebox.showerror("Error", "No face detected!")
        return

    x, y, w_box, h_box = latest_bbox
    face_img = latest_frame[y:y+h_box, x:x+w_box]
    if face_img.size == 0:
        messagebox.showerror("Error", "Failed to crop face!")
        return

    face_resized = cv2.resize(face_img, (128, 128))
    face_array = face_resized.astype("float32") / 255.0
    face_array = np.expand_dims(face_array, axis=0)
    new_embedding = embedding_model.predict(face_array)
    distance = np.linalg.norm(stored_embedding - new_embedding)
    threshold = 0.35

    if distance < threshold:
        result = f"Verification Successful: Distance = {distance:.4f}"
    else:
        result = f"Verification Failed: Distance = {distance:.4f}"
    status_label.config(text=result)
    messagebox.showinfo("Verification", result)

def show_embeddings():
    top = tk.Toplevel(root)
    top.title("Stored Embeddings")
    top.geometry("400x300")

    if not embeddings_dict:
        tk.Label(top, text="No embeddings stored.").pack(padx=10, pady=10)
        return

    for name in list(embeddings_dict.keys()):
        frame = tk.Frame(top)
        frame.pack(fill="x", padx=5, pady=2)
        tk.Label(frame, text=name).pack(side="left", padx=5)
        del_btn = tk.Button(frame, text="Delete", command=lambda n=name: delete_embedding(n, top))
        del_btn.pack(side="right", padx=5)

def delete_embedding(name, window):
    if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete embedding for {name}?"):
        del embeddings_dict[name]
        save_embeddings()
        window.destroy()
        show_embeddings()

def exit_app():
    root.destroy()

# --- Create UI Buttons ---
button_frame = tk.Frame(root)
button_frame.pack(pady=5)

toggle_button = tk.Button(button_frame, text="Toggle Face Detection", command=toggle_detection)
toggle_button.pack(side=tk.LEFT, padx=5)

live_view_button = tk.Button(button_frame, text="Toggle Live View", command=toggle_live_view)
live_view_button.pack(side=tk.LEFT, padx=5)

capture_button = tk.Button(button_frame, text="Capture Face and Save Embedding", command=capture_face)
capture_button.pack(side=tk.LEFT, padx=5)

verify_button = tk.Button(button_frame, text="Verify Face", command=verify_face)
verify_button.pack(side=tk.LEFT, padx=5)

show_button = tk.Button(button_frame, text="Show All Embeddings", command=show_embeddings)
show_button.pack(side=tk.LEFT, padx=5)

exit_button = tk.Button(button_frame, text="Exit Application", command=exit_app)
exit_button.pack(side=tk.LEFT, padx=5)

# --- Start the Video Loop and Tkinter Main Loop ---
update_frame()
root.mainloop()

# Release video capture when closing the UI
cap.release()
