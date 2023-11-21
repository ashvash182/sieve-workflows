import sieve
import random


@sieve.function(name='video-thumbnails',
                python_packages=["openai", "moviepy", "sentence_transformers", "Pillow", "numpy"],
                system_packages=["ffmpeg"],
                run_commands=["mkdir -p /src/fonts", "git clone https://github.com/ashvash182/workflow-custom-fonts /src/fonts"])
def main(video : sieve.Video, video_title : str, font : str, CLIP_prompts : str):
    '''
    :param video: A video input
    :param video_title: Desired video title, if one is not provided it will be generated from the transcript
    :param font: Select from Bebas_Neue (Default), Oswald, or Montserrat
    :param CLIP_prompts: A comma-separated list of prompts for objects, people, actions, etc. desired in the final thumbnail
    :return: A set of video thumbnails
    '''

    from moviepy.editor import VideoFileClip
    from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

    import concurrent
    import shutil
    import os
    from sentence_transformers import SentenceTransformer, util
    from PIL import Image
    import time
    import numpy as np

    # Load functions
    optimal_text_overlay = sieve.function.get('sieve-internal/optimal_text_overlay')
    side_text_overlay = sieve.function.get('sieve-internal/side_text_overlay')
    custom_scendetect = sieve.function.get('sieve-internal/custom_pyscenedetect')
    get_video_title = sieve.function.get('sieve-internal/generate_video_title')

    print('loading CLIP...')

    # Load CLIP
    model = SentenceTransformer('clip-ViT-L-14')
    img_model = SentenceTransformer('clip-ViT-L-14')

    # Subdirs for split videos and keyframes
    os.makedirs("./subvideos", exist_ok=True)
    os.makedirs("./scene_outputs/", exist_ok=True)
    
    print('extracting scene keyframes...')

    t = time.time()

    video_clip = VideoFileClip(video.path)
    video_duration = video_clip.duration
    video_mins = video_duration//60

    # How many videos to processin parallel, based on input video length
    if video_mins < 3:
        n_subvideos = 1
    elif video_mins < 5:
        n_subvideos = 2
    elif video_mins < 10:
        n_subvideos = 3
    else:
        n_subvideos = 8

    subvideo_duration = video_duration / n_subvideos

    subclips = []

    for i in range(n_subvideos):
        start_time = i * subvideo_duration
        end_time = (i + 1) * subvideo_duration
        output_video_file = f"./subvideos/subvideo_{i + 1}.mp4"
        subclips.append(sieve.Video(path=output_video_file))
        ffmpeg_extract_subclip(video.path, start_time, end_time, targetname=output_video_file)

    scenes = []

    if video_mins > 10:
        frame_skip = 12
    else:
        frame_skip = 8

    # Concurrent keyframe extraction with scene detect
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for job in executor.map(custom_scendetect.push, subclips, [frame_skip]*len(subclips)):
            res = list(job.result())
            if res:
                scenes.append(res)

    print(f"Total time to extract keyframes: {time.time() - t} seconds")

    scenes = list(scenes)

    # Get video title if not provided
    if not video_title:
        video_title = get_video_title.run(video).replace("\"", "")
    if not font:
        font = "Bebas_Neue"

    # if not scenes:
    #     return

    CLIP_outputs = []

    # Function to query over embedded frames
    def search(query, k=5):
        query_emb = model.encode([query], convert_to_tensor=True, show_progress_bar=False)

        hits = util.semantic_search(query_emb, img_emb, top_k=k)[0]
        
        for hit in hits:
            image_path = img_names[hit['corpus_id']]  
            CLIP_outputs.append(sieve.Image(path=image_path))
                    
    img_names = [img[0].path for img in scenes]

    print('creating image embeddings...')

    # Image embedding
    img_emb = img_model.encode([Image.open(img[0].path) for img in scenes], batch_size=128, convert_to_tensor=True, show_progress_bar=True)

    print('searching for prompts...')

    # Index as <= 3 prompts
    for prompt in CLIP_prompts.split(","):
        search(prompt, k=3)

    if not CLIP_outputs:
        print('No CLIP inputs provided')
        return
    
    print('creating thumbnails...')
    
    font_path = f'/src/fonts/{font}/{font.replace("_","")}-Regular.ttf'

    print('font path ', font_path)

    # Side text overlay output
    yield side_text_overlay.run(np.random.choice(CLIP_outputs), np.random.choice(CLIP_outputs), video_title, font_path)

    # Concurrent processing text simple text overlay
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for job in executor.map(optimal_text_overlay.push, CLIP_outputs, [video_title]*len(CLIP_outputs), [font_path]*len(CLIP_outputs)):
            if job.result() is not None:
                yield from job.result()

    print('finished!')

    if os.path.exists('./scene_outputs'):
        shutil.rmtree('./scene_outputs')
    if os.path.exists('./subvideos'):
        shutil.rmtree('./subvideos')

# if __name__ == "__main__":
#     print('run?')
#     main(video=sieve.Video(path="./videos/muskcolbert.mp4"), font="Oswald", CLIP_prompts="stephen colbert, elon musk, talkshow")
#     print('done')