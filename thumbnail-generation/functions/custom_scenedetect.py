import sieve

@sieve.function(name="custom_pyscenedetect",
                python_packages=[
                    "scenedetect[opencv]",
                    "opencv-python-headless==4.5.5.64",
                    "moviepy"],
                system_packages=["libgl1", "ffmpeg"])
def custom_pyscenedetect(video : sieve.Video, frame_skip : int):
    from scenedetect import open_video, SceneManager, ContentDetector, AdaptiveDetector, save_images
    
    from moviepy.editor import VideoFileClip

    import tempfile
    import os

    video = open_video(video.path)
    scene_manager = SceneManager()
    scene_manager.auto_downscale = True
    scene_manager.add_detector(AdaptiveDetector())
    
    output_dir = './scene_outputs/'

    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    
    video_clip_mins = VideoFileClip(video.path).duration/60
        
    if not frame_skip and video_clip_mins > 5:
        frame_skip = 12

    # frame_skip = 4
    
    # if (video_clip_mins <= 5):
    #     frame_skip = 0

    scene_manager.detect_scenes(video, frame_skip=frame_skip)
    scenes = scene_manager.get_scene_list()
    
    save_images(scenes, video=video, num_images=1, output_dir=output_dir, image_extension='jpg')

    files = os.listdir(output_dir)
    
    # Filter the files to include only image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    image_files = [file for file in files if os.path.splitext(file)[-1].lower() in image_extensions]
    
    # Yield the file paths
    for image_file in image_files:
        yield sieve.Image(path=os.path.join(output_dir, image_file))