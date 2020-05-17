"""
Created on May 15 2020

@author: neongreen13

Example:
    python audio_detection_bjjmatch.py


"""
import warnings
warnings.filterwarnings('ignore')
import argparse

import youtube_dl
from moviepy.editor import *
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.video.io.ffmpeg_tools import ffmpeg_movie_from_frames
from moviepy.editor import VideoFileClip, concatenate_videoclips

import librosa
import speech_recognition as sr
import IPython.display as ipd
import imageio
import cv2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import time
import datetime as dt

from os import getcwd
import glob
import os
from natsort import natsorted
import sys
from os.path import isfile
import json

# ap = argparse.ArgumentParser()
# ap.add_argument("--url", "-u", action='append',required=True)
# # ap.add_argument("--path", "-p", help="/home/neongreen13/riggs/BJJ-Position-Classification-master/bjj_videos/full_vids_predictions/test/test_args/")
# ap.add_argument("-path", "--path", type=str,
# 	help="/home/neongreen13/riggs/BJJ-Position-Classification-master/bjj_videos/full_vids_predictions/test/test_args/")
# # ap.add_argument("--output_path", "-o", type=str, help="/home/neongreen13/riggs/BJJ-Position-Classification-master/bjj_videos/full_vids_predictions/test/test_args/")
# args = vars(ap.parse_args())
#
# def create_arg_parser():
#     """"Creates and returns the ArgumentParser object."""
#
#     parser = argparse.ArgumentParser(description='Description of your app.')
#     parser.add_argument('inputDirectory',
#                     help='/home/neongreen13/riggs/BJJ-Position-Classification-master/bjj_videos/full_vids_predictions/test/test_args/')
#     parser.add_argument('--outputDirectory',
#                     help='/home/neongreen13/riggs/BJJ-Position-Classification-master/bjj_videos/full_vids_predictions/test/test_args/')
# return parser
#
#

path = '/home/neongreen13/riggs/BJJ-Position-Classification-master/bjj_videos/full_vids_predictions/test/test_args/'
url = 'https://www.youtube.com/watch?v=x0l-FroEc0c'


def download_url(url):

    options = {}
    os.chdir(path)
    with youtube_dl.YoutubeDL(options) as ydl:
        ydl.download([url])
    videoname = [os.path.basename(x) for x in glob.glob(path + '*.mp4')]
    return videoname

def video_to_audio(videoname):

    videoclip = VideoFileClip(path + videoname[0])
    audioclip = videoclip.audio
    audioclip.write_audiofile(path + "match_audio.wav")
    audio_file = [os.path.basename(x) for x in glob.glob(path + '*.wav')]
    return audio_file

def audio_energy(audio_file):

    filename = path + audio_file[0]
    x, sr = librosa.load(filename,sr=16000)
    int(librosa.get_duration(x, sr)/60)
    max_slice=5
    window_length = max_slice * sr
    energy = np.array([sum(abs(x[i:i+window_length]**2)) for i in range(0, len(x), window_length)])
    energy_90 = np.percentile(energy, 90)
    df=pd.DataFrame(columns=['energy','start','end'])
    thresh = energy_90
    row_index=0
    for i in range(len(energy)):
        value=energy[i]
        if(value>=thresh):
            i=np.where(energy == value)[0]
            df.loc[row_index,'energy']=value
            df.loc[row_index,'start']=i[0] * 5
            df.loc[row_index,'end']=(i[0]+1) * 5
            row_index= row_index + 1
    temp=[]
    i=0
    j=0
    n=len(df) - 2
    m=len(df) - 1
    while(i<=n):
        j=i+1
        while(j<=m):
            if(df['end'][i] == df['start'][j]):
                df.loc[i,'end'] = df.loc[j,'end']
                temp.append(j)
                j=j+1
            else:
                i=j
                break
    df.drop(temp,axis=0,inplace=True)
    df.to_csv(path + 'energy_file.csv',index=False)

    return df

def make_clips(df,videoname):

    clips_path = path + '/clips_folder/'
    if os.path.isdir(clips_path) == False:
        os.mkdir(clips_path)

    df = pd.read_csv(path + 'energy_file.csv')
    start=np.array(df['start'])
    end=np.array(df['end'])
    for i in range(len(df)):
        if(i!=0):
            start_lim = start[i] - 5
        else:
            start_lim = start[i]
        end_lim   = end[i]
        filename="highlight" + str(i+1) + ".mp4"
        clip = VideoFileClip(path + videoname[0]).subclip(start_lim,end_lim)
        clip.write_videofile(clips_path + filename)

    return clips_path

def clips_to_highlight():

    clips_path = path + '/clips_folder/'
    L =[]
    for root, dirs, files in os.walk(clips_path):

        files = natsorted(files)
        for file in files:
            if os.path.splitext(file)[1] == '.mp4':
                filePath = os.path.join(root, file)
                video = VideoFileClip(filePath)
                L.append(video)

    final_clip = concatenate_videoclips(L)
    final_clip.to_videofile(path + "match_highlights.mp4", fps=24, remove_temp=False)

    return True

def main(path,url):

    videoname = download_url(url)
    audio_file = video_to_audio(videoname)
    df = audio_energy(audio_file)
    make_clips(df,videoname)
    clips_to_highlight()

    return True

if __name__ == "__main__":
    try:
        finished = main(path,url)
    except FileNotFoundError:
        print("The file does not exist in the current working directory.")
        exit()
    if finished:
        print("\nDone.")
