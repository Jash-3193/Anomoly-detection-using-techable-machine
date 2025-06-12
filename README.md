Bottle Cap Defect Detector (Streamlit App)
This project is a deep learning web app that detects if a bottle is Good (with cap) or Defective (without cap) using an image. Built using TensorFlow, Keras, and Streamlit, it allows real-time image classification through a user-friendly web interface.

 How It Works
Upload an image of a bottle (with or without a cap).

The model analyzes the image using a trained CNN.

The app displays:

Predicted Status: Good / Defective
Confidence Score
A simple recommendation based on the result.

Model Info
Trained using Teachable Machine (Image Classification).

Exported as TensorFlow SavedModel.

Input shape: (224, 224, 3)

Two output classes:
0 Good
1 Defective
