import sieve
import random


@sieve.function(name='video-thumbnails',
                python_packages=["openai", "sentence_transformers", "Pillow"],
                system_packages=["ffmpeg"],
                run_commands=["mkdir -p /src/fonts", "git clone https://github.com/ashvash182/workflow-custom-fonts /src/fonts"])
def main(video : sieve.Video, font : str, CLIP_prompts : str):
    '''
    :param video: A video input
    :param font: Desired font for video title (See https://github.com/ashvash182/workflow-custom-fonts), or let LLM auto-select based on video style
    :param CLIP_prompts: A comma-separated list of prompts for objects, people, actions, etc. desired in the final thumbnail
    :return: A set of video thumbnails
    '''
    import shutil
    import os
    import tempfile
    import openai
    from sentence_transformers import SentenceTransformer, util
    from PIL import Image

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

    # IMPLEMENT CONCURRENCY
    scenes = sieve.function.get('ansh-sievedata-com/custom_pyscenedetect').run(video, 12)

    scenes = list(scenes)

    if not scenes:
        # Output singular video frame, meaning no scenes extracted!
        return
                
    video_title = sieve.function.get('ansh-sievedata-com/generate_video_title').run(video).replace("\"", "")

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
                    
    img_names = [img.path for img in scenes]

    print('creating image embeddings...')

    img_emb = img_model.encode([Image.open(img.path) for img in scenes], batch_size=128, convert_to_tensor=True, show_progress_bar=True)

    if os.path.exists('./scene_outputs'):
        import shutil
        shutil.rmtree('./scene_outputs')

    print('searching for prompts...')

    for prompt in CLIP_prompts.split(","):
        search(prompt, k=5)

    if not CLIP_outputs:
        CLIP_outputs = scenes
    
    print('creating thumbnails...')
    
    optimal_text_overlay = sieve.function.get('ansh-sievedata-com/optimal_text_overlay')

    font_path = f'/src/fonts/{font}/{font.replace("_","")}-Regular.ttf'

    print('font path ', font_path)

    # IMPLEMENT CONCURRENCY
    for out in CLIP_outputs:
        yield from optimal_text_overlay.run(out, video_title, font_path)

    print('finished!')

# if __name__ == "__main__":
#     print('run?')
#     main(video=sieve.Video(path="./videos/muskcolbert.mp4"), font="Oswald", CLIP_prompts="stephen colbert, elon musk, talkshow")
#     print('done')