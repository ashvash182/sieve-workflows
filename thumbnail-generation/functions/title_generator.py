import sieve

@sieve.function(name="generate_video_title",
                python_packages=[
                    "moviepy", 
                    "openai"],
                environment_variables=[sieve.Env(name="OPENAI_API_KEY", description="OpenAI API Key")])
def generate_title(video : sieve.Video):
  import os
  import openai
  from openai import OpenAI
  from moviepy.editor import VideoFileClip

  client = OpenAI()

  vid_path = video.path

  audio_path = vid_path.replace(".mp4", ".mp3")

  video = VideoFileClip(vid_path)

  video.audio.write_audiofile(audio_path)

  audio_file= open(audio_path, "rb")

  print('extracting video transcript...')

  transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file, response_format="text")

  print('creating video title...')

  # FIX ISSUE WITH LONG TRANSCRIPTS EXCEEDING TOKEN LIMIT
  response = client.chat.completions.create(
    model = "gpt-3.5-turbo-0613",
    messages=[
        {'role': 'system', 'content': 'create a short, simple title for the video this transcript is from. no quotes in response.'},
        {'role': 'user', 'content': transcript}
    ]
  )

  os.remove(audio_path)

  output = response.choices[0].message.content
  
  return output