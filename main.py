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
    embedding_model = load_model("face_recognition.h5")

# --- Initialize MediaPipe Face Detection ---
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# --- Global Variables ---
detection_enabled = False       # flag for face detection
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

    # If face detection is enabled, process and draw the bounding box
    if detection_enabled:
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
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        else:
            latest_bbox = None

    # Convert frame to RGB and display in Tkinter
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    root.after(10, update_frame)

# --- Button Callback Functions ---
def toggle_detection():
    global detection_enabled
    detection_enabled = not detection_enabled
    if detection_enabled:
        status_label.config(text="Status: Face Detection Enabled")
    else:
        status_label.config(text="Status: Face Detection Disabled")

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

    # Preprocess the face image as required by your model
    face_resized = cv2.resize(face_img, (128, 128))
    face_array = face_resized.astype("float32") / 255.0
    face_array = np.expand_dims(face_array, axis=0)

    # Generate the embedding from the model
    embedding = embedding_model.predict(face_array)
    
    # Get user name from entry
    user_name = name_entry.get().strip()
    if user_name == "":
        messagebox.showerror("Error", "Please enter your name!")
        return

    # Store the embedding (convert numpy array to list for serialization)
    embeddings_dict[user_name] = embedding.tolist()

    # Save updated embeddings to file
    save_embeddings()

    status_label.config(text=f"Status: Embedding saved for {user_name}")

def verify_face():
    global latest_frame, latest_bbox, embeddings_dict
    # Retrieve the stored name from entry
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

    # Crop the face from the current frame using the latest bounding box
    x, y, w_box, h_box = latest_bbox
    face_img = latest_frame[y:y+h_box, x:x+w_box]
    if face_img.size == 0:
        messagebox.showerror("Error", "Failed to crop face!")
        return

    # Preprocess the face image
    face_resized = cv2.resize(face_img, (128, 128))
    face_array = face_resized.astype("float32") / 255.0
    face_array = np.expand_dims(face_array, axis=0)

    # Create a new embedding from the current face
    new_embedding = embedding_model.predict(face_array)

    # Calculate Euclidean distance between the stored and new embeddings
    distance = np.linalg.norm(stored_embedding - new_embedding)
    threshold = 0.35  # Distance threshold for verification

    if distance < threshold:
        result = f"Verification Successful: Distance = {distance:.4f}"
    else:
        result = f"Verification Failed: Distance = {distance:.4f}"
    status_label.config(text=result)
    messagebox.showinfo("Verification", result)

def show_embeddings():
    # Create a new window to display all stored embeddings
    top = tk.Toplevel(root)
    top.title("Stored Embeddings")
    # Set the window size to be larger (e.g., 800x600 pixels)
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
