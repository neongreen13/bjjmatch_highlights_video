# bjjmatch_highlights_video
Extract audio clips from bjj sports match to create a highlights video

Process:
1. Find sport match on YouTube, provide url link and set path.
2. Download video in mp4 format to local computer.
3. Extract an audio file from video in a wav format.
4. Divide audio into energy clips throughout audio:
  - Segment audio into 5 sec intervals.
  - Compute short time energy for each 5 sec clip.
  - Set threshold of energy spike to all audio above 90% of energy level.
  - Return a dataframe with the energy level for each spike and the start and end times in the video.
5. Cut the video according to the audio clips set by the df.
6. Combine all clips into one highlights video for viewing.

Learning Resources:

- https://www.analyticsvidhya.com/blog/2019/09/guide-automatic-highlight-generation-python-without-machine-learning/
- https://www.analyticsvidhya.com/blog/2019/07/learn-build-first-speech-to-text-model-python/

