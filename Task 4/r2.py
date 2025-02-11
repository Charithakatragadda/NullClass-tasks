import cv2
import numpy as np
import pandas as pd
import datetime
import os
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt
from tensorflow.keras.models import load_model

# Load the trained model
MODEL_PATH = r"C:\Users\chari\Desktop\Age and Gender Detector\Age-Gender-Detector\TASK3\Age and Gender Detection1.keras"
model = load_model(MODEL_PATH)

class AgeGenderDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎢 Roller Coaster Age Detection System 🎡")
        self.setGeometry(100, 100, 500, 600)
        self.setStyleSheet("background-color: #f5f5f5;")
        
        # Title Label
        title_label = QLabel("🎠 Upload Image for Entry Check 🎠", self)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font: bold 20px 'Arial'; color: #333; padding: 10px;")

        # Image Display Label
        self.image_label = QLabel("No Image Uploaded", self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(300, 300)
        self.image_label.setStyleSheet("border: 2px dashed #aaa; background-color: #ffffff;")

        # Result Display Label
        self.result_label = QLabel("", self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font: bold 16px 'Arial'; color: green; padding: 5px;")

        # Upload Button
        self.upload_button = QPushButton("📁 Upload Image and Predict 🎲")
        self.upload_button.setStyleSheet("background-color: #0078d7; color: white; font: bold 14px; padding: 10px; border-radius: 8px;")
        self.upload_button.clicked.connect(self.upload_and_predict)

        # Layout Setup
        layout = QVBoxLayout()
        layout.addWidget(title_label)
        layout.addWidget(self.image_label)
        layout.addWidget(self.result_label)
        layout.addWidget(self.upload_button)
        layout.setAlignment(Qt.AlignCenter)

        self.setLayout(layout)

    def upload_and_predict(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Upload Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            self.predict_and_display(file_path)

    def predict_and_display(self, file_path):
        image = cv2.imread(file_path)
        image_resized = cv2.resize(image, (48, 48))
        image_normalized = image_resized.astype('float32') / 255.0
        image_expanded = np.expand_dims(image_normalized, axis=0)

        # Prediction
        predictions = model.predict(image_expanded)
        age = int(predictions[1][0])
        gender = "Female" if predictions[0][0] > 0.5 else "Male"

        is_not_allowed = age < 13 or age > 60
        status_text = "🚫 Not Allowed" if is_not_allowed else "✅ Allowed"

        # Update labels
        self.result_label.setStyleSheet(
            "color: red;" if is_not_allowed else "color: green;"
        )
        self.result_label.setText(f"{status_text}\nAge: {age}, Gender: {gender}")
        self.image_label.setPixmap(self.display_image_with_box(image, is_not_allowed))

        # Save data to CSV
        self.save_data(age, gender, is_not_allowed)

    def display_image_with_box(self, image, is_not_allowed):
        image_display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if is_not_allowed:
            height, width, _ = image.shape
            cv2.rectangle(image_display, (10, 10), (width - 10, height - 10), (255, 0, 0), 5)
        qimage = QImage(image_display.data, image_display.shape[1], image_display.shape[0],
                         image_display.strides[0], QImage.Format_RGB888)
        return QPixmap.fromImage(qimage).scaled(300, 300, Qt.KeepAspectRatio)

    def save_data(self, age, gender, is_not_allowed):
        file_name = 'senior_citizens_log.csv'
        columns = ['Age', 'Gender', 'Status', 'Entry Time']
        if os.path.exists(file_name):
            data = pd.read_csv(file_name)
        else:
            data = pd.DataFrame(columns=columns)
        entry_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_data = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'Status': "Not Allowed" if is_not_allowed else "Allowed",
            'Entry Time': entry_time
        }])

        data = pd.concat([data, new_data], ignore_index=True)
        data.to_csv(file_name, index=False)


if __name__ == "__main__":
    app = QApplication([])
    window = AgeGenderDetectionApp()
    window.show()
    app.exec_()
