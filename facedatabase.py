import cv2
import os

class FaceDetector:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


    def detect_faces(self):
        # Create the output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        # Loop over each image in the input directory
        for filename in os.listdir(self.input_dir):
            # Skip non-image files
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue

            # Load the input image
            img_path = os.path.join(self.input_dir, filename)
            img = cv2.imread(img_path)

            # Convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Crop the faces and save them in the output directory
            for i, (x, y, w, h) in enumerate(faces):
                face_img = img[y:y+h, x:x+w]
                output_path = os.path.join(self.output_dir, f"{os.path.splitext(filename)[0]}.jpg")
                cv2.imwrite(output_path, face_img)

            # # Display the original image with detected faces
            # for (x, y, w, h) in faces:
            #     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            #
            # # Save the image with detected faces in the input directory
            # output_path = os.path.join(self.input_dir, f"{os.path.splitext(filename)[0]}_detected.jpg")
            # cv2.imwrite(output_path, img)

        cv2.destroyAllWindows()

# Initialize the face detector object
fd = FaceDetector(input_dir='faces', output_dir='cropped_faces')

# Call the detect_faces method to detect faces in the input images and save the cropped faces in the output directory
fd.detect_faces()