### **Automated Unknown Person Detection and Email Image Notification System**

This project is an intelligent surveillance solution titled **"Automated Unknown Person Detection and Email Image Notification System"**, developed to enhance real-time monitoring and security. Using deep learning-based object detection (SSD MobileNet) and facial recognition, the system continuously scans video feeds to identify human presence. It compares detected faces against a dataset of known individuals. If an unknown person is detected, it automatically captures the frame, highlights the individual, and sends an email alert with the attached image. The system features a user-friendly GUI built with Tkinter that displays system status and real-time detection feed. It is ideal for use in homes, offices, schools, or any environment requiring automated intrusion detection and notification.

---

## 🔧 Features
- Real-time video stream monitoring
- Detection of persons using SSD MobileNet v2
- Face recognition using `face_recognition` module
- Email alert system with attachment
- GUI using Tkinter for status updates and live feedback
- Auto cleanup of detection images to save space
- Color-coded confidence display (Green for known, Red for unknown)

## 📦 Tech Stack / Modules Used
- Python
- OpenCV (`cv2`)
- face_recognition
- NumPy
- Tkinter (GUI)
- smtplib (email sending)
- SSD MobileNet v2 COCO model

## 🛠️ How It Works
1. The script initializes by loading known faces and pre-trained object detection models.
2. It opens a video stream (camera or file).
3. Each frame is scanned for people using the object detection model.
4. For each detected person, face recognition compares it to the known faces.
5. If the face is unknown:
   - The face is saved as an image
   - An email is sent with the captured image
   - GUI and terminal logs are updated

## 📂 Folder Structure
```
project_root/
├── known_faces/             # Contains images of known persons
├── person_detector/objects/ # Stores temporary detection frames
├── ssd_mobilenet_v2_coco_2018_03_29/
│   └── frozen_inference_graph.pb
├── new_labels.txt           # COCO labels file
├── ssd_mobilenet.pbtxt.txt  # Model config file
├── send_mail.py             # Handles email sending logic
└── person_detector_video.py # Main execution file
```

## 🚀 Run Instructions
```bash
python person_detector_video.py \
  --video 0 \
  --from_mail_id "youremail@gmail.com" \
  --pwd "yourapppassword" \
  --to_mail_id "receiver1@gmail.com" "receiver2@gmail.com"
```

## 📸 Output Preview
- Live camera feed with bounding boxes
- GUI window showing detection status
- Terminal log with color-coded system info
- Email with image of unknown person

## 📌 Use Cases
- Home security
- Office surveillance
- Classroom/hostel monitoring
- ATM booth safety
- Warehouse entry monitoring

## ✅ Benefits
- Automated real-time alerting
- No manual monitoring needed
- Lightweight, runs on basic hardware
- Easily customizable with new faces or improved models

## 🧠 Developed By
**Dontharaveni Rajender**  
B.Tech CSE | Vaagdevi College of Engineering, Warangal

---

> Feel free to fork, contribute, or integrate this system with advanced security platforms. Let's build safer spaces with AI! 🚀
