import cv2
import threading
import time
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

class VisionEngine:
    def __init__(self, db_manager=None, user_id=None):
        self.db = db_manager
        self.user_id = user_id
        
        # Load Haar Cascades
        self.face_cascade = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        )
        self.eye_cascade = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")
        )
        self.smile_cascade = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_smile.xml")
        )

        # Classifier
        self.classifier = None
        self.is_ml_trained = False
        
        # Calibration state
        self.is_calibrating = False
        self.calibration_target = None
        self.calibration_buffer = []
        
        # Running state
        self.running = False
        self.cap = None
        self.thread = None
        self.current_frame = None
        
        # Real-time metrics
        self.latest_metrics = {
            "emotion": "Focused",
            "confidence": 1.0,
            "focus_score": 100.0,
            "blink_count": 0,
            "fatigue_alert": False
        }
        
        # Internal focus tracking states
        self.blink_accumulator = 0
        self.eye_closed_frames = 0
        self.no_face_frames = 0
        
        # Try to train classifier on startup if calibration data exists
        self.retrain_classifier()

    def retrain_classifier(self):
        """
        Attempts to load calibration data from MongoDB and train a scikit-learn classifier.
        """
        if not self.db or not self.user_id:
            self.is_ml_trained = False
            return False
            
        data = self.db.get_all_calibration_data(self.user_id)
        if not data or len(data) < 2:  # Need at least 2 distinct emotions to classify
            print("Vision Engine: Insufficient calibration data. Rule-based model active.")
            self.is_ml_trained = False
            return False
            
        X = []
        y = []
        for record in data:
            emotion = record["emotion"]
            features = record["features"]
            for f in features:
                if len(f) == 6:
                    X.append(f)
                    y.append(emotion)
                    
        if len(set(y)) < 2:
            print("Vision Engine: Calibration data contains only one class. Rule-based model active.")
            self.is_ml_trained = False
            return False

        try:
            self.classifier = RandomForestClassifier(n_estimators=50, random_state=42)
            self.classifier.fit(X, y)
            self.is_ml_trained = True
            print(f"Vision Engine: scikit-learn classifier trained successfully on {len(X)} samples across classes: {list(set(y))}")
            return True
        except Exception as e:
            print(f"Error training classifier: {e}")
            self.is_ml_trained = False
            return False

    def start_calibration(self, emotion):
        """
        Enables recording of feature vectors for a specific emotion.
        """
        self.calibration_target = emotion
        self.calibration_buffer = []
        self.is_calibrating = True
        print(f"Vision Engine: Started calibrating for emotion '{emotion}'")

    def stop_and_save_calibration(self):
        """
        Saves the recorded calibration features to MongoDB and retrains the classifier.
        """
        self.is_calibrating = False
        if not self.db or not self.user_id or not self.calibration_target:
            return False
            
        if len(self.calibration_buffer) > 0:
            success = self.db.save_calibration_data(self.user_id, self.calibration_target, self.calibration_buffer)
            if success:
                print(f"Vision Engine: Saved {len(self.calibration_buffer)} samples for '{self.calibration_target}'")
                self.retrain_classifier()
                return True
        return False

    def extract_features(self, gray_frame):
        """
        Extracts 6 normalized facial geometry features using Haar Cascades.
        """
        faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))
        if len(faces) == 0:
            return None, None
            
        # Get largest face by area
        face = max(faces, key=lambda f: f[2] * f[3])
        (x, y, w, h) = face
        
        # Feature 0: Face Aspect Ratio (w / h)
        face_ar = float(w) / float(h)
        
        # Region of interest for eyes (upper 55% of the face)
        eye_roi_y = int(y + h * 0.1)
        eye_roi_h = int(h * 0.45)
        eye_roi_gray = gray_frame[eye_roi_y:eye_roi_y+eye_roi_h, x:x+w]
        
        eyes = self.eye_cascade.detectMultiScale(eye_roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(15, 15))
        
        eyes_count = len(eyes)
        eye1_ratio = 0.0
        eye2_ratio = 0.0
        
        if eyes_count > 0:
            # Sort eyes by relative x-coordinate
            sorted_eyes = sorted(eyes, key=lambda e: e[0])
            eye1_ratio = float(sorted_eyes[0][2]) / float(w)
            if eyes_count > 1:
                eye2_ratio = float(sorted_eyes[1][2]) / float(w)
                
        # Region of interest for smile (lower 40% of the face)
        smile_roi_y = int(y + h * 0.55)
        smile_roi_h = int(h * 0.4)
        smile_roi_gray = gray_frame[smile_roi_y:smile_roi_y+smile_roi_h, x:x+w]
        
        smiles = self.smile_cascade.detectMultiScale(smile_roi_gray, scaleFactor=1.3, minNeighbors=8, minSize=(20, 20))
        
        smile_detected = 1.0 if len(smiles) > 0 else 0.0
        smile_ratio = 0.0
        if smile_detected == 1.0:
            # Get largest smile
            smile = max(smiles, key=lambda s: s[2] * s[3])
            smile_ratio = float(smile[2]) / float(w)
            
        # Construct feature vector of length 6
        features = [
            face_ar,       # Feature 0
            eye1_ratio,    # Feature 1
            eye2_ratio,    # Feature 2
            smile_detected,# Feature 3
            smile_ratio,   # Feature 4
            float(eyes_count) # Feature 5
        ]
        return features, face

    def start(self):
        if self.running:
            return
        self.running = True
        self.cap = cv2.VideoCapture(0)
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print("Vision Engine background thread started.")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        self.cap = None
        self.current_frame = None
        print("Vision Engine background thread stopped.")

    def _capture_loop(self):
        last_log_time = time.time()
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.03)
                continue
                
            # Flip image horizontally for natural mirroring
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            features, face_box = self.extract_features(gray)
            
            # Predict or evaluate state
            current_emotion = "Focused"
            confidence = 1.0
            focus_score = 100.0
            fatigue_alert = False
            
            if face_box is not None:
                self.no_face_frames = 0
                (x, y, w, h) = face_box
                
                # Draw facial bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 100), 2)
                
                # Feature values check
                eyes_detected = features[5]
                smile_detected = features[3]
                
                # Handle Blink and Fatigue Tracking
                if eyes_detected == 0:
                    self.eye_closed_frames += 1
                else:
                    # Blink detection: eyes closed briefly then opened
                    if 1 <= self.eye_closed_frames <= 3:
                        self.blink_accumulator += 1
                    self.eye_closed_frames = 0
                    
                if self.eye_closed_frames > 15: # Eyes closed for ~0.5s continuously
                    fatigue_alert = True
                    current_emotion = "Tired"
                    focus_score = 30.0
                    cv2.putText(frame, "FATIGUE ALERT!", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Handle Emotion Inference
                if not fatigue_alert:
                    if self.is_calibrating:
                        self.calibration_buffer.append(features)
                        cv2.putText(frame, f"Calibrating: {self.calibration_target} ({len(self.calibration_buffer)})", 
                                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                    
                    if self.is_ml_trained and self.classifier:
                        # Machine Learning Mode
                        try:
                            pred = self.classifier.predict([features])[0]
                            probs = self.classifier.predict_proba([features])[0]
                            pred_idx = list(self.classifier.classes_).index(pred)
                            confidence = float(probs[pred_idx])
                            current_emotion = pred
                            
                            # Adjust focus score based on emotion
                            if current_emotion == "Focused":
                                focus_score = 90.0 + (confidence * 10)
                            elif current_emotion == "Happy":
                                focus_score = 80.0
                            elif current_emotion == "Stressed":
                                focus_score = 40.0 + (1.0 - confidence) * 20
                        except Exception as e:
                            print(f"Error predicting ML: {e}")
                    else:
                        # Fallback Rule-Based Inference
                        if smile_detected == 1.0:
                            current_emotion = "Happy"
                            confidence = 0.80
                            focus_score = 85.0
                        elif self.blink_accumulator > 6: # High blink rate indicates stress
                            current_emotion = "Stressed"
                            confidence = 0.70
                            focus_score = 50.0
                        else:
                            current_emotion = "Focused"
                            confidence = 0.90
                            focus_score = 95.0
                
                # Render HUD text on screen
                color = (0, 255, 100) if current_emotion == "Focused" else (255, 165, 0)
                if current_emotion == "Stressed":
                    color = (255, 0, 100)
                elif current_emotion == "Tired":
                    color = (0, 0, 255)
                    
                cv2.putText(frame, f"{current_emotion} ({confidence:.2f})", 
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                # No face detected
                self.no_face_frames += 1
                if self.no_face_frames > 20: # No face for ~0.6s
                    current_emotion = "Tired"
                    focus_score = 10.0
                    fatigue_alert = True
                else:
                    current_emotion = self.latest_metrics["emotion"]
                    focus_score = max(10.0, self.latest_metrics["focus_score"] - 5)
                    
            # Decay blink accumulator periodically (every 10 seconds, reset count)
            if time.time() - last_log_time >= 10.0:
                # Log metrics to database every 10 seconds
                if self.db and self.user_id and face_box is not None:
                    self.db.log_mood(
                        self.user_id, 
                        current_emotion, 
                        confidence, 
                        focus_score, 
                        self.blink_accumulator
                    )
                self.latest_metrics["blink_count"] = self.blink_accumulator
                self.blink_accumulator = 0
                last_log_time = time.time()

            # Update latest variables
            self.latest_metrics["emotion"] = current_emotion
            self.latest_metrics["confidence"] = confidence
            self.latest_metrics["focus_score"] = focus_score
            self.latest_metrics["fatigue_alert"] = fatigue_alert
            
            # Save frame for GUI fetching
            # Convert frame color space for Tkinter rendering
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_frame = rgb_frame
            
            # Control frame rate (~30 FPS)
            time.sleep(0.033)
            
    def get_latest_frame(self, target_width=320, target_height=240):
        """
        Resizes and returns the current frame as a PIL Image.
        """
        if self.current_frame is None:
            return None
            
        try:
            # Resize using PIL for compatibility
            pil_img = Image.fromarray(self.current_frame)
            pil_img = pil_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            return pil_img
        except Exception as e:
            print(f"Error resizing frame: {e}")
            return None
