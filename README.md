# UROP

The goal of this project was to detect hands in a video and track them over the timeframe of a hand stroke or gesture. The program generates images to represent the movement of the hands over the course of a stroke. This was written in Java, so you will need a Java compiler as well as a downloaded copy of OpenCV in order to get this to run. To run this, simply download a copy and open it in a Java compiler. You will want to run HandDetection.java as this is the main program. Additionally, this uses the OpenCV face recognition feature, so you will need to locate the classifier file and change the path to that file in the code. You will also need to create an account on faceplusplus.com and acquire your own key and add that to the code, and finally you will need to save a video file and a CSV file listing the strokes in a 2xn grid where the first column is start time in seconds and the second is end time in seconds. The algorithm currently works in the following order:

Iterate over the frames in a stroke, doing the following:
Detect human figures
Within those figures, cancel out all pixels that are clearly not normal skin tones (blue, green, etc)
Detect a face on the figure
Use the color palette of the face to search for similarly colored pixels that could be hands
Reduce noise by cutting down loose pixels
Create a final image and color it in shades of blue to red depending on when the hands are at what times during the strokes.
Repeat as many times as there are strokes listed in the CSV file

The user-changeable variables are located at the top of HandDetection.java and are:

histogramCompression: this is the compression rate for generating the 3D chart of skin tone HSV values. 8 is standard, although 4 or 16 might have different effects and be useful for higher/lower quality videos. Should probably remain as a power of 2, but maybe another value will work better.

compressionRate: this is the factor by which each frame will be compressed as a whole in each dimension. Generally, this doesn’t affect overall quality very much, but makes the program run much faster.


videoPath: file path to the video

timeChartLocation: path to CSV file with stroke times

cascadeClassifierLocation: path to the haar cascade classifier file

savePath: path to the save file of the final images of each stroke. Stop and start time of each stroke will be appended to the name automatically to prevent overwrites

tempSavePath: location to save, send and receive temporary images from faceplusplus.com

There are some things that I would have liked to have implemented given more time. First, this program would be faster and better if there was a way to directly detect hands versus arms/legs/etc. Second, there is still some amount of noise, so a more efficient way to get rid of that would be helpful. It also isn’t clear whether the output frames are ideal for visualizing strokes, perhaps a better visualization method would help. Finally, I had hoped to create a way to automatically detect hand velocity or speed over the course of a stroke, but did not have time to do that. Right now the program only outputs visual/qualitative data, but I think it would not be too hard to have it output quantitative data about the stroke.
