import numpy as np
import os
from sentence_transformers import SentenceTransformer, util
from PIL import Image, ImageDraw, ImageFont
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import argparse
import sieve

fonts_api_key = 'AIzaSyDA6LUiRHWFMwluojjn-_T6WPxH-NTOOzY'
    
# Query embedded images
def query(model, img_emb, img_names, query, k=9):
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
        
# Create title from video contents
def generate_title(video):
    video = sieve.Video(path=video)
    get_title = sieve.function.get('sieve/video_transcript_analyzer')
    return get_title.run(video)

# Get face BB's to guide text placement on image
def face_coords(img):
    img = sieve.Image(path=img)
    get_coords = sieve.function.get('sieve/mediapipe_video_face_detector')
    return get_coords.run(sieve.Image(path=img))
    
# Start regular pipeline
def main(model, img_model, vid_name, title_text, num_frames, title_size):
    if not os.path.exists('./scene-keyframes') or not any(os.listdir('./scene-keyframes')):
        subprocess.run(['scenedetect', '-i', vid_name, 'save-images', '-o', './scene-keyframes'])

    data_path = './scene-keyframes/'

    img_names = list(glob.glob(f'{data_path}*.jpg'))

    # img_emb = img_model.encode([Image.open(filepath) for filepath in img_names], batch_size=128, convert_to_tensor=True, show_progress_bar=True)

    # Will use generate_title here
    title = generate_title(vid_name)

    width = 512
    height = 512

    font = ImageFont.truetype("./Lato/Lato-Black.ttf", size=20)

    # imgDraw = ImageDraw.Draw(img)

    # imgDraw.text((10, 10), title, font=font, fill=(255, 255, 0))

    # img.save('result.png')

    # query(model, img_emb, img_names, prompt, num_frames)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--video", type=str, help="Video path")
    parser.add_argument("--num_frames", default=4, type=int, help="Number of thumbnails to return")
    # https://github.com/Nirvan101/Person-Re-identification
    parser.add_argument("--title_text", default="", help="Presence of title, and if specified or needs to be generated")
    parser.add_argument("--text_size", default=0.5, type=int, help="Size of title text, in between 0-1")
    # parser.add_argument("--graphic", default=None, type=str, help="Option to include graphic in thumbnail, e.g. text, art)

    args = parser.parse_args()
    
    # Initialize models
    model = SentenceTransformer('clip-ViT-B-32-multilingual-v1')
    img_model = SentenceTransformer('clip-ViT-B-32')

    main(model, img_model, args.video, args.title_text, args.num_frames, args.text_size)