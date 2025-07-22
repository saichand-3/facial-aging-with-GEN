from argparse import Namespace
import os
import sys
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

sys.path.append(".")
sys.path.append("..")

from datasets.augmentations import AgeTransformer
from utils.common import tensor2im
from models.psp import pSp

# Force device to CPU since CUDA is unavailable
device = 'cpu'

EXPERIMENT_TYPE = 'fhq_aging'
EXPERIMENT_DATA_ARGS = {
    "fhq_aging": {
        "model_path": "pretrained_models/sam_ffhq_aging.pt",
        "image_path": "notebooks/images/image6.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }
}

EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[EXPERIMENT_TYPE]
model_path = EXPERIMENT_ARGS['model_path']
ckpt = torch.load(model_path, map_location=device)

# Load model options and override device
opts = ckpt['opts']
opts['checkpoint_path'] = model_path
opts['device'] = device  # 💡 important!
opts = Namespace(**opts)

net = pSp(opts)
net.eval()
net.to(device)  # ✅ moved to CPU
print('Model successfully loaded on CPU!')
image_path = EXPERIMENT_ARGS["image_path"]

#image_path = EXPERIMENT_ARGS[EXPERIMENT_TYPE]["image_path"]
original_image = Image.open(image_path).convert("RGB").resize((256, 256))

def run_alignment(image_path):
    import dlib
    from scripts.align_all_parallel import align_face
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat.bz2.1.out")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image

aligned_image = run_alignment(image_path)
img_transforms = EXPERIMENT_ARGS['transform']
input_image = img_transforms(aligned_image)

target_ages = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
age_transformers = [AgeTransformer(target_age=age) for age in target_ages]

def run_on_batch(inputs, net):
    result_batch = net(inputs.to(device).float(), randomize_noise=False, resize=False)
    return result_batch

results = np.array(aligned_image.resize((1024, 1024)))

for age_transformer in age_transformers:
    print(f"Running on target age: {(age_transformer.target_age)+10}")
    with torch.no_grad():
        input_image_age = [age_transformer(input_image.cpu()).to(device)]
        input_image_age = torch.stack(input_image_age)
        result_tensor = run_on_batch(input_image_age, net)[0]
        result_image = tensor2im(result_tensor)
        results = np.concatenate([results, result_image], axis=1)

results = Image.fromarray(results)
results.show()  # optionally display the result
results.save("output_aging_result1.jpg")  # or save the result

#graphs
import matplotlib.pyplot as plt
import numpy as np
ages=[0,10,20,30,40,50,60,70,80,90,100]
def display_labeled_timeline(image_array, ages,save_path="timeline1.png"):
    num_faces = len(ages) + 1  # +1 for original
    img_width = image_array.width // num_faces
    img_height = image_array.height

    plt.figure(figsize=(20, 4))
    for i in range(num_faces):
        face = image_array.crop((i*img_width, 0, (i+1)*img_width, img_height))
        plt.subplot(1, num_faces, i + 1)
        plt.imshow(face)
        plt.axis('off')
        title = "Original" if i == 0 else f"Age {(ages[i-1])+10}"
        plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"1. Timeline graph saved to: {save_path}")

# Show the labeled timeline
display_labeled_timeline(results,ages)
#2
import numpy as np

def compute_pixel_diff(results_np, ages):
    original = results_np[:, :1024, :].astype(np.float32)  # First image
    diffs = []
    #for i in range(1, len(ages) + 1):
    #   aged = results_np[:, i*1024:(i+1)*1024, :].astype(np.float32)
    for i in range(len(ages)):
        aged = results_np[:, (i+1)*1024:(i+2)*1024, :]
        diff = np.mean(np.abs(original - aged))
        diffs.append(diff)
    return diffs

pixel_diffs = compute_pixel_diff(np.array(results), ages)

# Plot
save_path="pixel_difference1.png"
plt.figure(figsize=(8, 4))
plt.plot([10,20,30,40,50,60,70,80,90,100,110], pixel_diffs, marker='o')
#plt.title('Pixel-wise Difference from Original vs Age')
plt.xlabel('Age')
plt.ylabel('Average Pixel Difference')
plt.grid(True)
plt.savefig(save_path)
plt.show()
print(f"2. Pixel Difference graph saved to: {save_path}")
#3
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import cv2  # OpenCV for image processing

def preprocess_image(image):
    """Convert image to grayscale and flatten it into a 1D array for similarity computation."""
    image = image.convert("RGB")  # Make sure it's in RGB format
    image = np.array(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    return gray_image.flatten()  # Flatten to a 1D array

def calculate_similarity(image1, image2):
    """Calculate the cosine similarity between two images."""
    image1_vector = preprocess_image(image1)
    image2_vector = preprocess_image(image2)
    return cosine_similarity([image1_vector], [image2_vector])[0][0]

def plot_aging_similarity_graph(image, ages, save_path="aging_similarity_graph1.png"):
    """Plot the aging process graph (Similarity vs Age)."""
    base_face = image.crop((0, 0, image.size[0] // len(ages), image.size[1]))  # Assume first face is at age 0

    similarities = []
    for i, age in enumerate(ages):
        # Crop the corresponding face for each age
        left = i * (image.size[0] // len(ages))
        right = left + (image.size[0] // len(ages))
        face = image.crop((left, 0, right, image.size[1]))

        # Calculate similarity between base face and the current face
        similarity = calculate_similarity(base_face, face)
        similarities.append(similarity)

    # Plot similarity vs. age
    plt.figure(figsize=(8, 6))
    plt.plot(ages, similarities, marker='o', color='b', label='Face Similarity')
    #plt.title("Face Similarity vs Age", fontsize=14)
    plt.xlabel("Age", fontsize=12)
    plt.ylabel("Cosine Similarity", fontsize=12)
    plt.grid(True)
    plt.xticks(ages)  # Set x-ticks to the age values
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"3. Similarity graph saved to: {save_path}")

# Usage Example
# Assuming you already have the large stitched image: results (PIL image)
plot_aging_similarity_graph(results, ages)
#4
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import cv2

def plot_similarity_bar_graph(image, ages, save_path="similarity_bar_graph1.png"):
    """Plot the aging process similarity using a bar graph."""
    base_face = image.crop((0, 0, image.size[0] // len(ages), image.size[1]))  # Assume first face is at age 0

    similarities = []
    for i, age in enumerate(ages):
        left = i * (image.size[0] // len(ages))
        right = left + (image.size[0] // len(ages))
        face = image.crop((left, 0, right, image.size[1]))

        similarity = calculate_similarity(base_face, face)
        similarities.append(similarity)

    # Plotting Bar Graph
    plt.figure(figsize=(10, 6))
    plt.bar(ages, similarities, color='b', alpha=0.7, label='Face Similarity')
    #plt.title("Face Similarity vs Age (Bar Graph)", fontsize=14)
    plt.xlabel("Age", fontsize=12)
    plt.ylabel("Cosine Similarity", fontsize=12)
    plt.xticks(ages)  # Set x-ticks to the age values
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"4. Bar graph saved to: {save_path}")

# Usage
plot_similarity_bar_graph(results, ages)
#5
def plot_similarity_histogram(image, ages, save_path="similarity_histogram1.png"):
    """Plot the distribution of similarity values using a histogram."""
    base_face = image.crop((0, 0, image.size[0] // len(ages), image.size[1]))  # Assume first face is at age 0

    similarities = []
    for i, age in enumerate(ages):
        left = i * (image.size[0] // len(ages))
        right = left + (image.size[0] // len(ages))
        face = image.crop((left, 0, right, image.size[1]))

        similarity = calculate_similarity(base_face, face)
        similarities.append(similarity)

    # Plotting Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(similarities, bins=10, color='purple', alpha=0.7)
    #plt.title("Face Similarity Distribution (Histogram)", fontsize=14)
    plt.xlabel("Cosine Similarity", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"5. Histogram saved to: {save_path}")

# Usage
plot_similarity_histogram(results, ages)
#6
def plot_pixel_intensity(image, ages, save_path="pixel_intensity_changes1.png"):
    """Plot changes in pixel intensity (brightness) across different ages."""
    intensities = []

    for i, age in enumerate(ages):
        left = i * (image.size[0] // len(ages))
        right = left + (image.size[0] // len(ages))
        face = image.crop((left, 0, right, image.size[1]))

        # Convert to grayscale to analyze pixel intensity
        gray_face = face.convert("L")
        pixel_values = np.array(gray_face)
        average_intensity = np.mean(pixel_values)  # Mean pixel intensity

        intensities.append(average_intensity)

    # Plotting Histogram
    plt.figure(figsize=(10, 6))
    plt.bar(ages, intensities, color='purple', alpha=0.7, label='Pixel Intensity')
    #plt.title("Pixel Intensity (Brightness) vs Age", fontsize=14)
    plt.xlabel("Age", fontsize=12)
    plt.ylabel("Average Pixel Intensity", fontsize=12)
    plt.xticks(ages)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"6. Pixel intensity graph saved to: {save_path}")

# Usage
plot_pixel_intensity(results, ages)
#7
def plot_wrinkle_intensity(image, ages, save_path="wrinkle_intensity1.png"):
    """Detect wrinkles using edge detection and plot intensity changes over time."""
    wrinkle_intensity = []

    for i, age in enumerate(ages):
        left = i * (image.size[0] // len(ages))
        right = left + (image.size[0] // len(ages))
        face = image.crop((left, 0, right, image.size[1]))

        # Convert image to grayscale
        gray_face = cv2.cvtColor(np.array(face), cv2.COLOR_RGB2GRAY)

        # Apply edge detection to identify wrinkles
        edges = cv2.Canny(gray_face, threshold1=50, threshold2=150)
        wrinkle_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])  # Proportion of edges (wrinkles)

        wrinkle_intensity.append(wrinkle_density)

    # Plotting Bar Graph
    plt.figure(figsize=(10, 6))
    plt.bar(ages, wrinkle_intensity, color='brown', alpha=0.7, label='Wrinkle Density')
    #plt.title("Wrinkle Density vs Age", fontsize=14)
    plt.xlabel("Age", fontsize=12)
    plt.ylabel("Wrinkle Density", fontsize=12)
    plt.xticks(ages)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"7. Wrinkle intensity graph saved to: {save_path}")

# Usage
plot_wrinkle_intensity(results, ages)
#8
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load pre-trained Haar Cascade classifiers for eyes
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def get_eye_area(face):
    """Detect eyes in the face and return the area."""
    gray_face = cv2.cvtColor(np.array(face), cv2.COLOR_RGB2GRAY)
    eyes = eye_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5)

    # Calculate area as sum of individual eye rectangles
    eye_area = sum([w * h for (x, y, w, h) in eyes])
    return eye_area

def plot_facial_region_changes(image, ages, save_path="facial_region_changes1.png"):
    """Plot proportional changes in facial regions (eyes) over time."""
    eye_areas = []

    for i, age in enumerate(ages):
        left = i * (image.size[0] // len(ages))
        right = left + (image.size[0] // len(ages))
        face = image.crop((left, 0, right, image.size[1]))

        # Extract facial region (eyes only for now)
        eye_area = get_eye_area(face)
        eye_areas.append(eye_area)

    # Plotting Bar Graph for Eye Area Change
    plt.figure(figsize=(10, 6))
    plt.bar(ages, eye_areas, color='skyblue')
    #plt.title("Eye Area vs Age", fontsize=14)
    plt.xlabel("Age", fontsize=12)
    plt.ylabel("Eye Area (in pixels)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"8. Facial region changes saved to: {save_path}")

# Usage
plot_facial_region_changes(results, ages)
#9
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Assuming you have a function that extracts facial landmarks
def get_landmarks(face):
    """Placeholder function to extract landmarks."""
    # You can replace this with a real landmark detection model (e.g., Dlib or MediaPipe)
    # For now, returning random points for illustration
    return [(np.random.randint(0, face.size[0]), np.random.randint(0, face.size[1])) for _ in range(68)]

def plot_landmark_changes(image, ages, feature="x", save_path="landmark_changes1.png"):
    """Plot the change in a specific facial landmark (X or Y coordinate) over time."""
    changes = []
    for i, age in enumerate(ages):
        left = i * (image.size[0] // len(ages))
        right = left + (image.size[0] // len(ages))
        face = image.crop((left, 0, right, image.size[1]))

        landmark_positions = get_landmarks(face)  # Assuming `get_landmarks` gives (x, y) positions
        if feature == "x":
            positions = [landmark[0] for landmark in landmark_positions]
        elif feature == "y":
            positions = [landmark[1] for landmark in landmark_positions]

        changes.append(np.mean(positions))  # Average of all landmarks for a given age

    # Plotting the Line Graph
    plt.figure(figsize=(10, 6))
    plt.plot(ages, changes, marker='o', color='orange', label=f'{feature.upper()} Landmark Changes')
    #plt.title(f"Change in {feature.upper()} Landmark Positions Over Age", fontsize=14)
    plt.xlabel("Age", fontsize=12)
    plt.ylabel(f"{feature.upper()} Coordinate", fontsize=12)
    plt.xticks(ages)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"9. Landmark changes saved to: {save_path}")

# Usage
plot_landmark_changes(results, ages, feature="x")

#10
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# True target ages (from your code)
target_ages = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Define age buckets
def get_age_bucket(age):
    if age <= 20:
        return 0  # Young
    elif age <= 39:
        return 1  # Adult
    elif age <= 59:
        return 2  # Middle-aged
    else:
        return 3  # Senior

# Create true labels
true_labels = [get_age_bucket(age) for age in target_ages]

# Dummy predicted labels (assume model made some mistakes for demo)
predicted_labels = [0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3]  # Fake predictions for now

# Create confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Display labels
age_bucket_labels = ['Young (0-20)', 'Adult (21-39)', 'Middle-aged (40-59)', 'Senior (60+)']
save_path="confusion_matrix.png"
# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=age_bucket_labels)
disp.plot(cmap=plt.cm.Blues, values_format='d')
#plt.title('Confusion Matrix for SAM Facial Aging')
plt.xticks(rotation=45)
plt.savefig(save_path)
plt.show()
print(f"10. Confusion Matrix(demo) saved to: {save_path}")

#11
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet34
from PIL import Image

# Assume `results` is already loaded/generated (the big horizontal stitched image)
# Assume target_ages already defined
target_ages = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# --- Step 1: Split faces ---
num_faces = len(target_ages)
face_width = results.width // num_faces
faces = []

for i in range(num_faces):
    left = i * face_width
    right = (i + 1) * face_width
    face = results.crop((left, 0, right, results.height))
    faces.append(face)

# --- Step 2: Define bucketing function ---
def bucket_age(age):
    if age <= 20:
        return 0  # Young
    elif age <= 39:
        return 1  # Adult
    elif age <= 59:
        return 2  # Middle-aged
    else:
        return 3  # Senior

true_labels = [bucket_age(age) for age in target_ages]

# --- Step 3: Load small ResNet model for age prediction ---
age_model = resnet34(pretrained=True)
age_model.fc = torch.nn.Linear(age_model.fc.in_features, 1)  # Predict single value
age_model.eval()

def predict_age(face_image):
    transform_predict = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img = transform_predict(face_image).unsqueeze(0)
    with torch.no_grad():
        output = age_model(img)
        age = output.item()
    return max(0, min(100, age))

# --- Step 4: Predict ages ---
predicted_labels = []
for face in faces:
    estimated_age = predict_age(face)
    predicted_bucket = bucket_age(estimated_age)
    predicted_labels.append(predicted_bucket)

# --- Step 5: Build and plot confusion matrix ---
class_names = ['Young (0-20)', 'Adult (21-39)', 'Middle-aged (40-59)', 'Senior (60+)']

cm = confusion_matrix(true_labels, predicted_labels, labels=[0,1,2,3])
cm_normalized = confusion_matrix(true_labels, predicted_labels, labels=[0,1,2,3], normalize='true')

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, square=True)

#plt.title("Confusion Matrix for SAM Facial Aging", fontsize=16)
save_path="confusion_matrix.png"
plt.xlabel("Predicted Age Group", fontsize=14)
plt.ylabel("True Age Group", fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(save_path)
plt.show()
print(f"11. Confusion Matrix saved to: {save_path}")


