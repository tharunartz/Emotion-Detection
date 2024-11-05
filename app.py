from flask import Flask, render_template, Response, request, flash, jsonify, redirect
import cv2
import numpy as np
import librosa
from tensorflow.keras.models import model_from_json
import os
from moviepy.editor import VideoFileClip

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Ensure the uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Load emotion labels for speech model
emotion_labels = {
    0: 'anger',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

# Load face emotion model
face_model = model_from_json(open(r"C:\Users\THARUNARTZ\full_stack\base\model.json", "r").read())
face_model.load_weights(r'C:\Users\THARUNARTZ\full_stack\base\model.h5')

# Load speech emotion model
def load_model_from_json(model_json_path, model_weights_path):
    with open(model_json_path, 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights(model_weights_path)
    return model

speech_model = load_model_from_json(r'C:\Users\THARUNARTZ\full_stack\base\speech.json', r'C:\Users\THARUNARTZ\full_stack\base\speech.h5')

# Load Haar Cascade for face detection
face_haar_cascade = cv2.CascadeClassifier(r'C:\Users\THARUNARTZ\full_stack\base\haarcascade_frontalface_default.xml')

# Index route
@app.route('/')
def index():
    return render_template('index.html')

# Existing route for real-time face detection
@app.route('/face')
def face_detection():
    return render_template('face.html')

# Existing route for speech emotion detection
@app.route('/speech', methods=['GET', 'POST'])
def speech_detection():
    emotion_detected = None
    if request.method == 'POST':
        if 'audiofile' not in request.files:
            flash('No file part')
            return render_template('speech.html', emotion=emotion_detected)
        file = request.files['audiofile']
        if file.filename == '':
            flash('No selected file')
            return render_template('speech.html', emotion=emotion_detected)
        if file:
            filepath = os.path.join('uploads', file.filename)
            file.save(filepath)
            processed_audio = preprocess_audio(filepath)
            emotion_detected = make_prediction(speech_model, processed_audio)
            os.remove(filepath)  # Clean up the uploaded file
    return render_template('speech.html', emotion=emotion_detected)

# Route to render the video upload page
@app.route('/video')
def video_upload():
    return render_template('video.html')

# New route to handle video uploads for frame-by-frame emotion detection
@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'videofile' not in request.files:
        flash('No file part')
        return redirect(request.url)
    video_file = request.files['videofile']
    if video_file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if video_file:
        video_path = os.path.join('uploads', video_file.filename)
        video_file.save(video_path)

        # Process video for frame-by-frame emotion detection
        results = process_video(video_path)

        # Remove uploaded file after processing
        os.remove(video_path)

        return jsonify(results)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_results = []
    audio_emotion = None

    # Extract and analyze audio for emotion detection
    audio_path = extract_audio_from_video(video_path)
    audio_emotion = analyze_audio_from_video(audio_path)

    # Process each frame in video for face emotion detection
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Face emotion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_haar_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray / 255.0
            roi_gray = np.expand_dims(roi_gray, axis=0)
            roi_gray = np.expand_dims(roi_gray, axis=-1)

            prediction = face_model.predict(roi_gray)
            emotion_index = np.argmax(prediction)
            emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            face_emotion = emotions[emotion_index]

            frame_results.append({"frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)), "emotion": face_emotion})

    cap.release()
    os.remove(audio_path)  # Clean up extracted audio file

    return {"audio_emotion": audio_emotion, "frame_by_frame_emotions": frame_results}

def extract_audio_from_video(video_path):
    audio_path = 'uploads/extracted_audio.wav'
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)
    video.close()
    return audio_path

def analyze_audio_from_video(audio_path):
    processed_audio = preprocess_audio(audio_path)
    emotion_detected = make_prediction(speech_model, processed_audio)
    return emotion_detected

def preprocess_audio(file_path, sr=22050, n_mfcc=65):
    audio, sample_rate = librosa.load(file_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    mfccs_reshaped = mfccs_scaled.reshape((n_mfcc, 1))
    return np.expand_dims(mfccs_reshaped, axis=0)

def make_prediction(model, processed_audio):
    prediction = model.predict(processed_audio)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = emotion_labels[predicted_class]
    return predicted_label

# Video streaming route for real-time face emotion detection
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    cap = cv2.VideoCapture(0)  # Capture video from the webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_haar_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray / 255.0
            roi_gray = np.expand_dims(roi_gray, axis=0)
            roi_gray = np.expand_dims(roi_gray, axis=-1)

            prediction = face_model.predict(roi_gray)
            emotion_index = np.argmax(prediction)
            emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            emotion = emotions[emotion_index]

            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    app.run(debug=True)
