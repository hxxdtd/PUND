import warnings
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="7"
from facenet_pytorch import MTCNN

from os.path import abspath, dirname
celeb_path = os.path.join(
    dirname(abspath(__file__)),
    'celeb-detection-oss',
)
import sys
sys.path.append(celeb_path)
from model_training.helpers.labels import Labels
from model_training.helpers.face_recognizer import FaceRecognizer



mtcnn = MTCNN(margin=0.1, thresholds=[0.6, 0.7, 0.7], select_largest=False, device="cuda")
resources_path = os.path.join(celeb_path, 'examples', 'resources')
model_labels = Labels(resources_path=resources_path)
face_recognizer = FaceRecognizer(
        labels=model_labels,
        resources_path=resources_path,
        use_cuda=True
        )

def crop_face(img, scale=0.709):  # input type: np.array
    pil_img = Image.fromarray(img)
    hal_img = pil_img.resize(size=[s // 2 for s in pil_img.size])

    batch_boxes, batch_probs, batch_points = mtcnn.detect(hal_img, landmarks=True)
    batch_boxes, batch_probs, batch_points = mtcnn.select_boxes(batch_boxes, batch_probs, batch_points, hal_img, method="probability")
    if batch_boxes is not None:
        xmin, ymin, xmax, ymax = [int(b * 2) for b in batch_boxes[0, :]]
        w = xmax - xmin
        h = ymax - ymin

        size_bb = int(max(w, h) * scale)
        center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2

        # Check for out of bounds, x-y top left corner
        xmin = max(int(center_x - size_bb // 2), 0)
        ymin = max(int(center_y - size_bb // 2), 0)
        # Check for too big bb size for given x, y
        size_bb = min(pil_img.size[1] - xmin, size_bb)
        size_bb = min(pil_img.size[0] - ymin, size_bb)

        crop = img[ymin:ymin + size_bb, xmin:xmin + size_bb]
        save_img = Image.fromarray(crop).resize((224, 224))
        return crop
    else:
        return None

def calculate_id_score(image, id):
    image = np.array(image)
    face_images = crop_face(image)

    if face_images is None:
        return 0.0
    else:
        recog = face_recognizer.perform([np.array(face_images)])
        for r in range(len(recog[0][0])): # get the top 5 IDs with the highest scores 
            if id.replace(' ', '_') in str(recog[0][0][r][0]).lower():
                return float(recog[0][0][r][1])
        
        return 0.0
    