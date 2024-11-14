# %% [markdown]
# # Audio Feature Extraction
# 
# ### This file has the extraction of audio features from the dataset.The steps performed are:
# 
# 1. From the peaks of the audio fall and non-fall data plotted, it was observed that most of the peaks last for about one tenth of a second. 
# 2. The peaks are those where the sound strength or the amplitude is the maximum where the fall is likely to have occurred and if each peak lasts for one tenth of a second, the amplitude could be extracted for every one tenth of a second and saved. 
# 3. By the use of the python library librosa, the amplitude of the data at every tenth of a second is extracted and stored.
# 4. The extracted features are exported to a csv file

# %% [markdown]
# Import depedencies

# %%
from __future__ import print_function
import librosa
import librosa.display
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import os

# %% [markdown]
# Feature extraction for sample audio using librosa

# %%
y, sr = librosa.load(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/SisFall_audios/Non_Falls/D3.mp3")
print(sr)
duration = librosa.get_duration(y=y, sr=sr)
onset_env = librosa.onset.onset_strength(y=y, sr=sr,hop_length=2205,aggregate=np.median)
pd.set_option('display.max_columns', None)
df=pd.DataFrame(onset_env)
print(df)

# %% [markdown]
# Feature extraction for all falls and non falls audio using librosa

# %%
fall_list = os.listdir(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/Mono_audios/Falls")
count = 1

for i in fall_list:
    y, sr = librosa.load(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/Mono_audios/Falls/F"+str(count)+".wav")
    duration = librosa.get_duration(y=y, sr=sr)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr,hop_length=2205,aggregate=np.median)
    pd.set_option('display.max_columns', None)
    df=pd.DataFrame(onset_env)
    df.to_csv(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/audio_features/falls/F"+str(count)+".csv") 
    count = count + 1


# %%
fall_list = os.listdir(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/Mono_audios/Non_Falls")
count = 1

for i in fall_list:
    y, sr = librosa.load(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/Mono_audios/Non_Falls/D"+str(count)+".wav")
    duration = librosa.get_duration(y=y, sr=sr)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr,hop_length=2205,aggregate=np.median)
    pd.set_option('display.max_columns', None)
    df=pd.DataFrame(onset_env)
    df.to_csv(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/audio_features/non_falls/D"+str(count)+".csv") 
    count = count + 1


# %% [markdown]
# 1. After generating feature files, the csv files were combined and transposed  
# 2. This step was done manually.
# 3. The final dataset was created having 172 features and a label value
# 4. This gave a dataset having 34 rows( 19 not falls 15 falls) and 172  columns
# 5. Out of which around 40 features were 0 for all the samples, which were removed
# 6. The final data had 34 rows (samples) and 133 columns(features)
# 7. To this the labels (0 or 1) was added


