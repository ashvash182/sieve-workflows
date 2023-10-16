import sieve

@sieve.function(name="custom_pyscenedetect",
                python_packages=[
                    "scenedetect[opencv]",
                    "opencv-python-headless==4.5.5.64",
                    "moviepy"],
                system_packages=["libgl1", "ffmpeg"])
def custom_pyscenedetect(video : sieve.Video):
    from scenedetect import open_video, SceneManager, ContentDetector, save_images

    from moviepy.editor import VideoFileClip

    import tempfile
    import os

    video = open_video(video.path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())

    output_dir = os.path.join(tempfile.gettempdir(), './scene_outputs/')
    
    video_clip_mins = VideoFileClip(video.path).duration/60

    scene_manager.detect_scenes(video, frame_skip=2*int(video_clip_mins))
    scenes = scene_manager.get_scene_list()
    
    save_images(scenes, video=video, num_images=1, output_dir=output_dir, image_extension='png')

    files = os.listdir(output_dir)
    
    # Filter the files to include only image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    image_files = [file for file in files if os.path.splitext(file)[-1].lower() in image_extensions]
    
    # Yield the file paths
    for image_file in image_files:
        yield sieve.Image(path=os.path.join(output_dir, image_file))
    
    # Need to remove tmpdir