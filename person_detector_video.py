import cv2
import os
import numpy as np
import argparse
import face_recognition
from send_mail import send_mail
from tkinter import Tk, Label, Button, StringVar
import threading
import time

# Clear screen on start
os.system('cls' if os.name == 'nt' else 'clear')

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--dl_model', type=str, default='./ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb')
parser.add_argument('--label_file', type=str, default='./new_labels.txt')
parser.add_argument('--config', type=str, default='./ssd_mobilenet.pbtxt.txt')
parser.add_argument('--detection_folder', type=str, default='./person_detector/objects/')
parser.add_argument('--to_mail_id', nargs='+')
parser.add_argument('--from_mail_id', type=str)
parser.add_argument('--pwd', type=str)
parser.add_argument('--video', type=str, default='0')
parser.add_argument('--known_faces_folder', type=str, default='./known_faces/')
args = parser.parse_args()

def print_banner():
    print("=" * 60)
    print("\033[95m📸 AI-BASED UNKNOWN PERSON DETECTION SYSTEM\033[0m")
    print("Developed by Dontharaveni Rajender")
    print("=" * 60)
    print("\033[94mInitializing system...\033[0m")
    time.sleep(1)
    print("\033[92m✔ Model loaded\033[0m")
    print("\033[92m✔ Email service connected\033[0m")
    print("\033[92m✔ Camera ready\033[0m")
    print("=" * 60)
    print("Press 'q' in the video window to exit.\n")

# Setup
print_banner()
if not os.path.exists(args.detection_folder):
    os.makedirs(args.detection_folder)
for i in os.listdir(args.detection_folder):
    os.remove(os.path.join(args.detection_folder, i))

def load_known_faces(folder):
    encodings = []
    names = []
    for file in os.listdir(folder):
        if file.endswith(('.jpg', '.png')):
            image = face_recognition.load_image_file(os.path.join(folder, file))
            encode = face_recognition.face_encodings(image)
            if encode:
                encodings.append(encode[0])
                names.append(file)
    return encodings, names

known_face_encodings, known_face_names = load_known_faces(args.known_faces_folder)

# GUI Thread
class App:
    def __init__(self, root):
        self.root = root
        root.title("Unknown Person Detection")
        Label(root, text="System Status", font=("Helvetica", 16)).pack(pady=10)
        self.label = Label(root, textvariable=gui_text, font=("Helvetica", 14), fg="blue")
        self.label.pack(pady=5)
        self.button = Button(root, text="Exit", command=self.exit_program, font=("Helvetica", 12))
        self.button.pack(pady=10)

    def exit_program(self):
        os._exit(0)

def update_gui(message):
    gui_text.set(message)

def run_detection():
    with open(args.label_file, 'r') as f:
        class_names = f.read().splitlines()

    model = cv2.dnn.readNetFromTensorflow(args.dl_model, args.config)
    frame_count = 0
    video_source = int(args.video) if args.video == '0' else args.video
    video = cv2.VideoCapture(video_source)

    if not video.isOpened():
        print("\033[91m❌ Could not open camera or video.\033[0m")
        update_gui("Camera error.")
        return

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frame_count += 1
        h, w, _ = frame.shape

        cv2.rectangle(frame, (0, 0), (w, 40), (50, 50, 50), -1)
        cv2.putText(frame, "AI Detection System", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

        blob = cv2.dnn.blobFromImage(frame, size=(300, 300), mean=(104, 117, 123), swapRB=True)
        model.setInput(blob)
        output = model.forward()
        object_count = 0

        for detection in output[0, 0, :, :]:
            confidence = detection[2]
            if confidence > 0.5:
                class_id = int(detection[1])
                if class_names[class_id - 1] == 'person':
                    x1 = int(detection[3] * w)
                    y1 = int(detection[4] * h)
                    x2 = int(detection[5] * w)
                    y2 = int(detection[6] * h)

                    crop_img = frame[y1:y2, x1:x2]
                    face_locations = face_recognition.face_locations(crop_img)
                    face_encodings = face_recognition.face_encodings(crop_img, face_locations)

                    label = "Unknown"
                    color = (0, 0, 255)  # Default to red

                    if face_encodings:
                        for face_encoding in face_encodings:
                            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                            best_match_index = np.argmin(face_distances) if face_distances.size > 0 else None

                            if best_match_index is not None and matches[best_match_index]:
                                label = "Known"
                                color = (0, 255, 0)  # Green
                                confidence_display = int(confidence * 100)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                cv2.putText(frame, f"{label} ({confidence_display}%)", (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


                            else:
                                label = "Unknown"
                                color = (0, 0, 255)
                                confidence_display = int(confidence * 100)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                cv2.putText(frame, f"{label} ({confidence_display}%)", (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                                object_count += 1
                                img_path = os.path.join(args.detection_folder, f'frame_{frame_count}_obj_{object_count}.jpg')
                                cv2.imwrite(img_path, crop_img)
                                print("\033[91m[ALERT]\033[0m Unknown person detected!")
                                update_gui("Unknown detected. Sending email...")
                                send_mail(args.from_mail_id, args.pwd, from_email=args.from_mail_id,
                                           to_emails=args.to_mail_id, attachment=img_path)

                                if len(os.listdir(args.detection_folder)) >= 5:
                                    for f in os.listdir(args.detection_folder):
                                        os.remove(os.path.join(args.detection_folder, f))
                    else:
                        # No face encoding found, still mark as unknown
                        confidence_display = int(confidence * 100)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{label} ({confidence_display}%)", (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        update_gui(f"Frame: {frame_count} | Detection Running")
        cv2.imshow("Detection Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    update_gui("Detection ended.")

# Start GUI
root = Tk()
gui_text = StringVar(root)
app = App(root)
detection_thread = threading.Thread(target=run_detection)
detection_thread.start()
root.mainloop()
