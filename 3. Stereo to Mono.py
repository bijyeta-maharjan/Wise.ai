# %% [markdown]
# # Stereo to Mono
# 
# ### This file has the conversion of stereo audio to mono audio for sepctrogram generation.The steps performed are:
# 

# %% [markdown]
# Import Dependencies

# %%
from pydub import AudioSegment
import os

# %% [markdown]
# Create list containing names of falls and non falls audio files

# %%
fall_list = os.listdir(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/SisFall_audios/Falls")
non_fall_list = os.listdir(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/SisFall_audios/Non_Falls")

# %% [markdown]
# Use inbuilt functions to read the stereo audio file and convert it to mono audio file.

# %%
count = 1
for i in fall_list:
    sound = AudioSegment.from_mp3(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/SisFall_audios/Falls/F"+str(count)+".mp3")
    sound = sound.set_channels(1)
    sound.export(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/Mono_audios/Falls/F"+str(count)+".wav", format='wav')
    count = count + 1

# %%
count = 1
for i in non_fall_list:
    sound = AudioSegment.from_mp3(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/SisFall_audios/Non_Falls/D"+str(count)+".mp3")
    sound = sound.set_channels(1)
    sound.export(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/Mono_audios/Non_Falls/D"+str(count)+".wav", format='wav')
    count = count + 1


