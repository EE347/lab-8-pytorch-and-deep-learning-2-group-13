import cv2
import torch
from torchvision.transforms import functional as TF
from torchvision.models import mobilenet_v3_small

# Load the trained model
model = mobilenet_v3_small(weights=None, num_classes=2)
model.load_state_dict(torch.load('lab8/best_model.pth'))
model.eval()

# Check for CUDA availability and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Initialize the webcam
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (64, 64))
    image = TF.to_tensor(image).unsqueeze(0)
    return image

# Function to classify the face
def classify_face(image):
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Crop the face from the frame
        face = frame[y:y+h, x:x+w]
        
        # Preprocess the face for classification
        processed_face = preprocess_image(face)
        
        # Classify the face
        label = classify_face(processed_face)
        
        # Draw a rectangle around the face and label it
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        if label == 0:
            cv2.putText(frame, 'Teammate 1', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        else:
            cv2.putText(frame, 'Teammate 2', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()