import argparse
import sieve
import concurrent

@sieve.function(name='video-thumbnails',
                system_packages=["ffmpeg"])
def workflow(video : sieve.Video):
    # Custom scene detection to account for longer videos and to return image paths for further filtering
    
    print('extracting scene keyframes...')

    scenes = sieve.function.get('ansh-sievedata-com/custom_pyscenedetect').run(video)
        
    face_detector = sieve.function.get('sieve/mediapipe_face_detector')
    # text_overlay = sieve.function.get # Will be sieve function
    # video_title = sieve.function.get('sieve/video_transcript_analyzer').run(video)["title"] # Will be sieve function
    
    def process_scene(scene):
        job = face_detector.push(scene)
        res = list(job.result())
        if res and not res[1] == []:
            return scene.path

    print('choosing frames with people present...')

    bbox_valid = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for job in executor.map(process_scene, scenes):
            if job:
                yield sieve.Image(path=job)
        
    # for r in res:
    #     if r:
    #         yield sieve.Image(path=r)

    print('finished!')

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
    
#     parser.add_argument("--video", type=str, help="Video path")
#     # parser.add_argument("--num_frames", default=4, type=int, help="Number of thumbnails to return")
#     # parser.add_argument("--title_text", default="", help="Presence of title, and if specified or needs to be generated")

#     args = parser.parse_args()

#     out = workflow(sieve.Video(path=args.video))

#     for o in out:
#         print(o)