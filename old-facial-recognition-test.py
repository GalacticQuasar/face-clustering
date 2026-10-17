import cv2
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from torchvision import transforms


def center_crop(img, dim):
    width, height = img.shape[1], img.shape[0]

    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
    mid_x, mid_y = width // 2, height // 2
    cw2, ch2 = crop_width // 2, crop_height // 2
    crop_img = img[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]

    return crop_img  # OpenCV Image


def get_embeddings(cropped_img):
    transform = transforms.Compose([
        transforms.Resize((160, 160)),  # Resize image to 160x160
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(  # Normalize image
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return resnet(transform(cropped_img).unsqueeze(0)).detach()


# Welcome message
print("Welcome to FacialDetectionVideo. Keep in mind that the escape character is 'q'.")

# Initialize MTCNN for face detection
print("Initializing MTCNN...")
mtcnn = MTCNN()  # Use MTCNN(device='cuda') for GPU

# Load pretrained ResNet model for facial recognition
print("Loading InceptionResnetV1")
resnet = InceptionResnetV1(pretrained='casia-webface').eval()

# Initialize OpenCV VideoCapture
print("Initializing OpenCV VideoCapture...")
cam = cv2.VideoCapture(0)  # Replace 0 with video file to run on video file

print("RUNNING")

# Set image processing resolution
maxRes = min(cam.get(cv2.CAP_PROP_FRAME_WIDTH), cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
resolution = 160
# resolution = int(input(f"Please enter the processing resolution (cannot be over {maxRes}): "))
#
# while resolution > maxRes:
#     resolution = int(input(f"ERROR. Input processing resolution (cannot be over {maxRes}): "))

# Obtain reference face embeddings for comparison
input("SETUP VERIFICATION: Click enter to take a picture of the desired face to identify.")
_, opencv_img = cam.read()
opencv_img = center_crop(opencv_img, (min(opencv_img.shape[0], opencv_img.shape[1]), min(opencv_img.shape[0], opencv_img.shape[1])))

# Convert OpenCV image to PIL image
ref_img = Image.fromarray(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB))

# Downscale
multiplier = ref_img.width / resolution  # height and width are equal
downscaled_img = ref_img.resize(size=(resolution, resolution))

# Extract face embeddings
ref_boxes, _ = mtcnn.detect(downscaled_img)
ref_boxes = [list(map(lambda x: x * multiplier, box)) for box in ref_boxes]  # Upscale coordinates to map on original
reference_embeddings = get_embeddings(ref_img.crop(ref_boxes[0]))

while True:
    result, frame = cam.read()

    # Center square crop OpenCV Image
    frame = center_crop(frame, (min(frame.shape[0], frame.shape[1]), min(frame.shape[0], frame.shape[1])))

    # Convert OpenCV Image to PIL
    color_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(color_converted)

    # Downscale
    multiplier = pil_img.width / resolution  # height and width are equal
    downscaled_img = pil_img.resize(size=(resolution, resolution))

    # Detect faces in the image
    boxes, _ = mtcnn.detect(downscaled_img)

    # Resize boxes to map on original image size
    if boxes is not None:
        boxes = [list(map(lambda x: x * multiplier, box)) for box in boxes]

        for box in boxes:
            # Draw bounding boxes on the image
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (128, 128, 0), 2)

            # Extract face embeddings and compare
            embeddings = get_embeddings(pil_img.crop(box))

            distance = (embeddings - reference_embeddings).norm().item()
            if distance < 1:
                label = "Identified"
            else:
                label = "Unidentified"

            # Overlay face label
            frame = cv2.putText(frame, label, (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 128, 0), 1, cv2.LINE_AA)

    cv2.imshow('display', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up OpenCV Resources
cam.release()
cv2.destroyAllWindows()