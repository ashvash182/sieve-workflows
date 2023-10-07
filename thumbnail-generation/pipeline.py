import numpy as np
import os
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
from IPython.display import display
from IPython.display import Image as IPImage
import os
import numpy as np
import matplotlib.pyplot as plt
from scenedetect import detect, ContentDetector, split_video_ffmpeg, AdaptiveDetector
import subprocess
import argparse

model = SentenceTransformer('clip-ViT-B-32-multilingual-v1')
img_model = SentenceTransformer('clip-ViT-B-32')

# Helper to display images
def plot_images(images, query, n_row=3, n_col=3):
    _, axs = plt.subplots(n_row, n_col, figsize=(12, 12))
    axs = axs.flatten()
    for img, ax in zip(images, axs):
        ax.set_title(query)
        ax.imshow(img)
    plt.show()
    
# Query embedded images
def search(img_emb, img_names, query, k=9):
    query_emb = model.encode([query], convert_to_tensor=True, show_progress_bar=False)
    
    hits = util.semantic_search(query_emb, img_emb, top_k=k)[0]
    
    matched_images = []
    for index, hit in enumerate(hits):
        matched_images.append(Image.open(img_names[hit['corpus_id']]))
        image_path = img_names[hit['corpus_id']]
        image = Image.open(image_path)
        
        filename = f"query_{query}_{index + 1}.png"
        save_path = os.path.join("./", filename)
        image.save(save_path)
        
    # plot_images(matched_images, query)

# Create title from video contents
def generate_title(video):
    raise NotImplementedError

# Google API to create font .png => Overlay on final thumbnail candidates
def apply_font(text):
    raise NotImplementedError
    
# Start pipeline
def main(vid_name, prompt):
    vid_name = './videos/balrog.mp4'

    subprocess.run(['scenedetect', '-i', vid_name, 'save-images', '-o', './scene-keyframes'])

    data_path = './scene-keyframes/'

    img_names = list(glob.glob(f'{data_path}*.jpg'))

    img_emb = img_model.encode([Image.open(filepath) for filepath in img_names], batch_size=128, convert_to_tensor=True, show_progress_bar=True)

    search(img_emb, img_names, prompt)

    for filename in os.listdir('./scene-keyframes/'):
        file_path = os.path.join('./scene-keyframes/', filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Add command-line arguments
    parser.add_argument("video", type=int, help="Description of arg1")

# Original method - Take all frames and run image QA (composition, low-blur, etc.), then take keyframes => Too slow

# input_video_path = 'balrog.mp4'
# output_directory = 'sampled_frames'
# os.makedirs(output_directory, exist_ok=True)

# cap = cv2.VideoCapture(input_video_path)
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# sampler = 15

# frame_number = 0

# while True:
#     ret, frame = cap.read()
    
#     if not ret:
#         break

#     if frame_number % sampler == 0:
#         frame_filename = os.path.join(output_directory, f'frame_{frame_number:04d}.jpg')
#         cv2.imwrite(frame_filename, frame)
#         frame_number += 1
    
#     frame_number += 1

# cap.release()
# cv2.destroyAllWindows()