# %% [markdown]
# # Audio Extraction from Videos
# 
# ### This file has the extraction of audio files from the video dataset.The steps performed are:
# 
# 1. The dataset used here is the SisFall dataset containing 19 non falls and 15 falls 
# 2. Each of the video file is converted into audios using moviepy and stored into pre created directories

# %% [markdown]
# Import necessary dependencies

# %%
import moviepy
import moviepy.editor
import os

# %% [markdown]
# Create list containing names of falls and non falls video files

# %%
fall_list = os.listdir(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/SisFall_audios/Falls")
non_fall_list = os.listdir(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/SisFall_audios/Non_Falls")

# %%
fall_list

# %%
non_fall_list

# %% [markdown]
# Use inbuilt functions to read the video file and convert it to audio file. Audio files are stored in created directories

# %%
count = 1
for i in fall_list:
    video = moviepy.editor.VideoFileClip(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/SisFall_videos/SisFall_videos/Falls/F"+str(count)+".mp4")
    audio = video.audio
    audio.write_audiofile(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/SisFall_audios/Falls/F"+str(count)+".mp3")
    count = count +1

# %%
count = 1
for i in non_fall_list:
    video = moviepy.editor.VideoFileClip(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/SisFall_videos/SisFall_videos/Not_falls/D"+str(count)+".mp4")
    audio = video.audio
    audio.write_audiofile(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/SisFall_audios/Non_Falls/D"+str(count)+".mp3")
    count = count +1


