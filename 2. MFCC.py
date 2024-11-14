# %% [markdown]
# # EDA using MFCC
# 
# ### This file has the MFCC features generation and EDA using MFCC.
# 
# 1. For each of the audio files extracted, MFCC features are visualized to better understand the data
# 2. To identify falls using audio data, firstly it must be noted if there was a significant variation in the amplitude, or energy of the audio signal.
# 3. The MFCC graphs indicated the thirteen different coefficients that could describe the audio with precision.
# 4. It was found that there was a clear increase in energy during certain peaks in the fall audios which was not present in the non-fall audios.
# 

# %% [markdown]
# Import dependencies

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from python_speech_features import logfbank
import os

# %% [markdown]
# For each fall and not fall audio, MFCC coefficients vs time and MFCC feature amplitudes vs time are generated using python_speech_features module. 
# The generated image are stored.

# %%
fall_list = os.listdir(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/Mono_audios/Falls")
count = 1

for i in fall_list:
    (rate,sig) = wav.read(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/Mono_audios/Falls/F"+str(count)+".wav")
    mfcc_feat = mfcc(sig,rate)
    ig, ax = plt.subplots()
    mfcc_data= np.swapaxes(mfcc_feat, 0 ,1)
    cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower', aspect='auto')
    ax.set_title('MFCC')
    plt.savefig(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/MFCC/Falls/F"+str(count)+".png")
    plt.plot(mfcc_feat)
    plt.savefig(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/MFCC Features/Falls/F"+str(count)+".png")
    count = count + 1

# %%
non_fall_list = os.listdir(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/Mono_audios/Non_Falls")
count = 1

for i in non_fall_list:
    (rate,sig) = wav.read(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/Mono_audios/Non_Falls/D"+str(count)+".wav")
    mfcc_feat = mfcc(sig,rate)
    ig, ax = plt.subplots()
    mfcc_data= np.swapaxes(mfcc_feat, 0 ,1)
    cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower', aspect='auto')
    ax.set_title('MFCC')
    plt.savefig(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/MFCC/Non_falls/D"+str(count)+".png")
    plt.plot(mfcc_feat)
    plt.savefig(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/MFCC Features/Non_falls/D"+str(count)+".png")
    count = count + 1

# %% [markdown]
# For visualization purpose, one audio from fall and non fall are selected and the two graphs are generated
# 
# 1. In the first graph, MFCC coefficients are plotted vs time  
# 2. The y-axis indicates the number of coefficients and the x-axis indicates the time. 
# 3. Here the lower amplitudes are indicated by blue and the higher ones by shades of red and it is clear that there are different points of high and low amplitudes for each feature. 
# 4. In the second graph, an actual plot of the features using their amplitudes as the y-axis is plotted vs time as the x-axis. 
# 5. It is very difficult to get meaningful information about the maximum amplitude overall by dissecting the preexisting audio. 
# 6. But nevertheless, it is important to observe that amplitude is a very important feature for classification.
# 7. It was found that there was a clear increase in energy during certain peaks in the fall audios which was not present in the non-fallaudios. 
# 8. This certainly suggests that there is a clear indication of increase in energy, amplitude when the fall occurred. 

# %%
(rate,sig) = wav.read(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/Mono_audios/Non_Falls/D3.wav")
mfcc_features = mfcc(sig,rate)

ig, ax = plt.subplots()
mfcc_data= np.swapaxes(mfcc_features, 0 ,1)
cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower', aspect='auto')
ax.set_title('MFCC')

plt.show()
plt.plot(mfcc_features)
plt.show()

# %%
(rate,sig) = wav.read(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/Mono_audios/Falls/F2.wav")
mfcc_features = mfcc(sig,rate)

ig, ax = plt.subplots()
mfcc_data= np.swapaxes(mfcc_features, 0 ,1)
cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower', aspect='auto')
ax.set_title('MFCC')

plt.show()
plt.plot(mfcc_features)
plt.show()


