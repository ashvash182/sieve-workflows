import sieve
import random


@sieve.function(name='video-thumbnails',
                python_packages=["openai", "sentence_transformers", "Pillow"],
                system_packages=["ffmpeg"])
def main(video : sieve.Video, font : str, CLIP_prompts : str):
    '''
    :param video: A video input
    :param font: Desired font for video title (See https://github.com/ashvash182/workflow-custom-fonts)
    :param CLIP_prompts: A comma-separated list of CLIP prompts for objects, people, etc. desired in the final thumbnail
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

    scenes = sieve.function.get('ansh-sievedata-com/custom_pyscenedetect').run(video)

    scenes = list(scenes)

    if not scenes:
        # Output singular video frame, meaning no scenes extracted!
        return
                
    # print('getting video title...')

    video_title = sieve.function.get('ansh-sievedata-com/generate_video_title').run(video)

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
    
    print('creating thumbnails...')

    text_overlay = sieve.function.get('ansh-sievedata-com/image_text_overlay')
    # text_overlay_two = sieve.function.get('ansh-sieve-data/image_text_overlay_two')

    font_path = f'/root/fonts/workflow-custom-fonts/{font}/{font.replace("_","")}-Regular.ttf'

    print('selected font ', font_path)

    for i in range(5):
        base, left, right = random.sample(CLIP_outputs, 3)
        yield text_overlay.run(base, left, right, video_title, font_path)
    # yield text_overlay_two.run(base, left, right, video_title, font_path)

    print('finished!')

if __name__ == "__main__":
    print('run?')
    main(video=sieve.Video(path="./videos/muskcolbert.mp4"), font="Oswald", CLIP_prompts="stephen colbert, elon musk, talkshow")
    print('done')