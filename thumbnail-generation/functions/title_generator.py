import sieve

@sieve.function(name="generate_video_title",
                python_packages=[
                    "moviepy", 
                    "openai"],
                environment_variables=[sieve.Env(name="OPENAI_API_KEY", description="OpenAI API Key")])
def generate_title(video : sieve.Video):
  import os
  import openai
  from moviepy.editor import VideoFileClip

  vid_path = video.path

  audio_path = vid_path.replace(".mp4", ".mp3")

  video = VideoFileClip(vid_path)

  video.audio.write_audiofile(audio_path)

  audio_file= open(audio_path, "rb")

  print('extracting video transcript...')

  transcript = openai.Audio.transcribe("whisper-1", audio_file)['text']

  print('creating video title...')

  response = openai.ChatCompletion.create(
    model = "gpt-3.5-turbo-0613",
    messages=[
        {'role': 'system', 'content': 'create a short, simple title for the video this transcript is from.'},
        {'role': 'user', 'content': transcript}
    ]
  )

  os.remove(audio_path)

  output = response.choices[0].message.content
  
  return output