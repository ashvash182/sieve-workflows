import sieve
import random


@sieve.function(name='video-thumbnails',
                python_packages=["openai", "moviepy", "sentence_transformers", "Pillow"],
                system_packages=["ffmpeg"],
                run_commands=["mkdir -p /src/fonts", "git clone https://github.com/ashvash182/workflow-custom-fonts /src/fonts"])
def main(video : sieve.Video, font : str, CLIP_prompts : str):
    '''
    :param video: A video input
    :param font: Desired font for video title (See https://github.com/ashvash182/workflow-custom-fonts), or let LLM auto-select based on video style
    :param CLIP_prompts: A comma-separated list of prompts for objects, people, actions, etc. desired in the final thumbnail
    :return: A set of video thumbnails
    '''

    from moviepy.editor import VideoFileClip
    from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

    import concurrent
    import shutil
    import os
    import tempfile
    from sentence_transformers import SentenceTransformer, util
    from PIL import Image
    import time

    if not font:
        font = "Bebas_Neue"

    print('loading CLIP...')

    model = SentenceTransformer('clip-ViT-L-14')
    img_model = SentenceTransformer('clip-ViT-L-14')

    temp_dir = os.path.join(tempfile.gettempdir(), '/scene_outputs/')

    if os.path.exists(temp_dir) and os.path.isdir(temp_dir):
        # The directory exists, so you can remove it
        shutil.rmtree(temp_dir)

    # Custom scene detection to account for longer videos and to return image paths for further filtering
    
    print('extracting scene keyframes...')

    # IMPLEMENTING CONCURRENCY

    t = time.time()

    custom_scendetect = sieve.function.get('sieve-internal/custom_pyscenedetect')

    output_directory = "./subvideos"
    os.makedirs(output_directory, exist_ok=True)

    video_clip = VideoFileClip(video.path)
    video_duration = video_clip.duration
    video_mins = video_duration//60

    if video_mins < 5:
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

    frame_skip = None

    if video_mins > 10:
        frame_skip = 12
    else:
        frame_skip = 8

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for job in executor.map(custom_scendetect.push, subclips, [frame_skip]*len(subclips)):
            res = list(job.result())
            if res:
                scenes.append(res)
                
    if os.path.exists(output_directory) and os.path.isdir(output_directory):
        # The directory exists, so you can remove it
        shutil.rmtree(output_directory)

    print(f"Total time to extract keyframes: {time.time() - t} seconds")
    scenes = list(scenes)

    video_title = sieve.function.get('sieve-internal/generate_video_title').run(video).replace("\"", "")

    # if not scenes:
    #     return
                
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer('clip-ViT-L-14')
    img_model = SentenceTransformer('clip-ViT-L-14')

    CLIP_outputs = []

    def search(query, k=5):
        query_emb = model.encode([query], convert_to_tensor=True, show_progress_bar=False)

        hits = util.semantic_search(query_emb, img_emb, top_k=k)[0]
        
        for hit in hits:
            image_path = img_names[hit['corpus_id']]  
            CLIP_outputs.append(sieve.Image(path=image_path))
                    
    img_names = [img[0].path for img in scenes]

    print('creating image embeddings...')

    img_emb = img_model.encode([Image.open(img[0].path) for img in scenes], batch_size=128, convert_to_tensor=True, show_progress_bar=True)

    if os.path.exists('./scene_outputs'):
        import shutil
        shutil.rmtree('./scene_outputs')

    print('searching for prompts...')

    for prompt in CLIP_prompts.split(","):
        search(prompt, k=3)

    if not CLIP_outputs:
        CLIP_outputs = scenes
    
    print('creating thumbnails...')
    
    optimal_text_overlay = sieve.function.get('sieve-internal/optimal_text_overlay')

    font_path = f'/src/fonts/{font}/{font.replace("_","")}-Regular.ttf'

    print('font path ', font_path)

    # IMPLEMENT CONCURRENCY
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for job in executor.map(optimal_text_overlay.push, CLIP_outputs, [video_title]*len(CLIP_outputs), [font_path]*len(CLIP_outputs)):
            yield from job.result()

    print('finished!')

# if __name__ == "__main__":
#     print('run?')
#     main(video=sieve.Video(path="./videos/muskcolbert.mp4"), font="Oswald", CLIP_prompts="stephen colbert, elon musk, talkshow")
#     print('done')