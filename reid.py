import numpy as np
from glob import glob
import cv2

import onnxruntime as ort
import numpy as np
import cv2
images = glob("*.jpg")[:20]
import matplotlib.pyplot as plt
# Load the ONNX ReID model
onnx_model_path = "./models/resnet50_market1501_aicity156.onnx"  # change to your model path
session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # e.g., [1, 3, 256, 128]

def preprocess(img_path, input_shape):
    """Resize, normalize and prepare image for model input"""
    img = cv2.imread(img_path)
    img = cv2.resize(img, (input_shape[3], input_shape[2]))
    img = img[:, :, ::-1]  # BGR to RGB
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0  # CHW, normalize to [0,1]
    img = np.expand_dims(img, axis=0)  # add batch dimension
    return img

def get_embedding(img_path):
    input_tensor = preprocess(img_path, input_shape)
    outputs = session.run(None, {input_name: input_tensor})
    return outputs[0][0]  # remove batch dim

def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

# Example usage
# vec1 = get_embedding(images[0])
# vec2 = get_embedding(images[1])

# similarity = cosine_similarity(vec1, vec2)

mat = np.zeros((len(images), len(images)))
red_vecs = {k:get_embedding(k) for k in images}
for i, img1 in enumerate(images):
    for j, img2 in enumerate(images):
        # print(i, j)
        if i<=j:
            continue
        similarity = cosine_similarity(red_vecs[img1], red_vecs[img2])
        if similarity>0.85:
            print(img1, img2, similarity)
        mat[i,j] = similarity
            # print(f"Cosine Similarity: {similarity:.4f}")


# print(mat)

# cv2.imshow("Image",mat)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# for img_path in images:
#     img = cv2.imread(img_path)
#     img = cv2.resize(img, (1188,648))
#     cv2.imshow("Image",img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     break

    