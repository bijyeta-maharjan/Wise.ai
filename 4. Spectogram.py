# %% [markdown]
# # EDA using Spectogram
# 
# ### This file has the Spectogram generation and EDA using Spectogram.
# 
# 1. For each of the audio files extracted, Spectograms are generated to better understand the data.
# 2. Spectrograms are utilized for visually representing the amplitude/signal strength of a signal at varying frequencies. 
# 3. They plot the association between the frequency and time in an audio signal and could help identify if there were any clear peaks in the given data.
# 4. To visualize the results more clearly, a separate amplitude time graph was also plotted in addition to the spectrograms. 
# 5. The Spectrograms indicated more brightness in certain peaks in case of fall audio which was not present in non-fall audio.
# 6. The places where there was maximum brightness in the spectrogram was analogous to the maximum amplitude in the amplitude vs time graph.
# 7. This indicates that the amplitude or loudness has increased to the maximum at around the same time in both of the graphs and this time can be approximated to the time the fall occurred.
# 

# %% [markdown]
# Import dependencies

# %%
import matplotlib.pyplot as plot
from scipy.io import wavfile
import os

# %% [markdown]
# For visualization purpose, one audio from fall and non fall are selected and the two graphs - the spectrogram and its corresponding amplitude-time graph is plotted.
# 
# 1. A sample fall audio is taken and it has to be noted that there is a clear peak that can be seen right around the 14th second of the video. 
# 2. On looking at the spectrogram, the brightest color is at the same time, and it becomes more evident when it is compared to the amplitude-time graph above it shows that the amplitude or loudness has drastically increased at around the same time.
# 3. Such a peak does not occur in the non fall audio plotted below

# %%
samplingFrequency, signalData = wavfile.read(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/Mono_audios/Falls/F1.wav")
plot.subplot(211)
plot.title('Spectrogram of a wav file')
plot.plot(signalData)
plot.xlabel('Sample')
plot.ylabel('Amplitude')
plot.subplot(212)
plot.specgram(signalData,Fs=samplingFrequency)
plot.xlabel('Time')
plot.ylabel('Frequency')
plot.colorbar()
plot.show()

# %%
samplingFrequency, signalData = wavfile.read(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/Mono_audios/Non_Falls/D1.wav")
plot.subplot(211)
plot.title('Spectrogram of a wav file')
plot.plot(signalData)
plot.xlabel('Sample')
plot.ylabel('Amplitude')
plot.subplot(212)
plot.specgram(signalData,Fs=samplingFrequency)
plot.xlabel('Time')
plot.ylabel('Frequency')
plot.colorbar()
plot.show()

# %% [markdown]
# Graphs generated for all falls and non falls and seperate analysis was performed.

# %%
fall_list = os.listdir(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/Mono_audios/Falls")
count = 1

for i in fall_list:   
    samplingFrequency, signalData = wavfile.read(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/Mono_audios/Falls/F"+str(count)+".wav")
    plot.subplot(211)
    plot.title('Spectrogram of a wav file')
    plot.plot(signalData)
    plot.xlabel('Sample')
    plot.ylabel('Amplitude')
    plot.subplot(212)
    plot.specgram(signalData,Fs=samplingFrequency)
    plot.xlabel('Time')
    plot.ylabel('Frequency')
    plot.colorbar()
    plot.savefig(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/spectrograms/time_graphs/falls/F"+str(count)+".png")
    count = count + 1

# %%
fall_list = os.listdir(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/Mono_audios/Non_Falls")
count = 1

for i in fall_list:   
    samplingFrequency, signalData = wavfile.read(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/Mono_audios/Non_Falls/D"+str(count)+".wav")
    plot.subplot(211)
    plot.title('Spectrogram of a wav file')
    plot.plot(signalData)
    plot.xlabel('Sample')
    plot.ylabel('Amplitude')
    plot.subplot(212)
    plot.specgram(signalData,Fs=samplingFrequency)
    plot.xlabel('Time')
    plot.ylabel('Frequency')
    plot.colorbar()
    plot.savefig(r"/Users/bijyetamaharjan/Documents/Projects/fall_detection_audios/spectrograms/time_graphs/non_falls/D"+str(count)+".png")
    count = count + 1


