import cv2
import numpy as np
import os
import datetime
from flask import Flask, Response, render_template, request, jsonify, url_for
from flask_socketio import SocketIO
from keras.models import load_model
import google.generativeai as genai

app = Flask(__name__)
socketio = SocketIO(app)

# ตั้งค่า API Key สำหรับ Gemini
GEMINI_API_KEY = 'AIzaSyDMPBWVBNZOHxsF2IlbjTLREHEx6LR2NZ4'
genai.configure(api_key=GEMINI_API_KEY)

# ตรวจสอบว่ามีไฟล์โมเดลอยู่หรือไม่
model_path = r'C:\Users\kpaop\Downloads\Face Recognition System\Face Recognition System\keras_model.h5'
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    raise FileNotFoundError(f"Model file '{model_path}' not found.")

# ตรวจสอบว่ามีไฟล์ Haarcascade อยู่หรือไม่
face_cascade_path = r'C:\Users\kpaop\Downloads\Face Recognition System\Face Recognition System\haarcascade_frontalface_default.xml'
if os.path.exists(face_cascade_path):
    facedetect = cv2.CascadeClassifier(face_cascade_path)
else:
    raise FileNotFoundError(f"Face cascade file '{face_cascade_path}' not found.")

# สร้างไดเรกทอรีเพื่อเก็บใบหน้าที่ตรวจพบ
detected_faces_dir = r'C:\path\to\your\folder\detected_faces'  # เปลี่ยนที่อยู่โฟลเดอร์ที่ต้องการเก็บ
if not os.path.exists(detected_faces_dir):
    os.makedirs(detected_faces_dir)

already_alerted_faces = set()  # ใช้เก็บชื่อใบหน้าที่เตือนแล้ว
detected_faces_data = []

# กำหนด threshold สำหรับการคาดการณ์
threshold = 0.5

# รายชื่อบุคคลที่รู้จัก (ควรกำหนดให้ตรงตามโมเดลของคุณ)
known_persons = ["Worramate", "Nakarin", "Apisada"]

# ฟังก์ชันการดึงเฟรมจากกล้อง
def generate_frames():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ไม่สามารถเชื่อมต่อกล้องได้")
        socketio.emit('alert', {'name': 'ไม่สามารถเชื่อมต่อกล้องได้'})
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ไม่สามารถดึงภาพจากกล้องได้")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_roi = frame[y:y+h, x:x+w]
            
            face_array = cv2.resize(face_roi, (224, 224))
            face_array = face_array / 255.0
            face_array = np.expand_dims(face_array, axis=0)

            predictions = model.predict(face_array)
            predicted_name = "บุคคลไม่รู้จัก" if predictions[0][0] < threshold else known_persons[np.argmax(predictions[0])]
            confidence = np.max(predictions[0])

            if predicted_name not in already_alerted_faces:
                already_alerted_faces.add(predicted_name)

                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                detected_faces_data.append({
                    "name": predicted_name,
                    "timestamp": timestamp,
                    "confidence": confidence,
                })

                face_image_path = os.path.join(detected_faces_dir, f'{predicted_name}_{timestamp}.jpg')
                cv2.imwrite(face_image_path, face_roi)

                # ตรวจสอบชื่อที่ตรวจพบ และส่งสัญญาณเตือนตามบุคคลที่ตรวจพบ
                if predicted_name == "บุคคลไม่รู้จัก":
                    socketio.emit('alert', {
                        'name': 'ผู้บุกรุก',
                        'image_path': f"/detected_faces/{os.path.basename(face_image_path)}"
                    })
                elif predicted_name == "Worramate":
                    socketio.emit('alert', {
                        'name': 'Worramate',
                        'image_path': f"/detected_faces/{os.path.basename(face_image_path)}"
                    })
                elif predicted_name == "Nakarin":
                    socketio.emit('alert', {
                        'name': 'Nakarin',
                        'image_path': f"/detected_faces/{os.path.basename(face_image_path)}"
                    })
                elif predicted_name == "Apisada":
                    socketio.emit('alert', {
                        'name': 'Apisada',
                        'image_path': f"/detected_faces/{os.path.basename(face_image_path)}"
                    })

            cv2.putText(frame, f"{predicted_name} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html', faces=detected_faces_data)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ฟังก์ชันการใช้งานแชทบอทผ่าน Gemini API
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    print(f"Received message: {user_message}")

    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config={
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        },
    )

    chat_session = model.start_chat(
        history=[{'role': 'user', 'content': user_message}]
    )

    response = chat_session.send_message(user_message)
    print(f"Gemini response: {response}")

    gemini_reply = response.text if response else 'เกิดข้อผิดพลาดในการเชื่อมต่อกับ Gemini API'

    return jsonify({'reply': gemini_reply})

if __name__ == '__main__':
    socketio.run(app, debug=True)
