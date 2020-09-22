

## How to use:

1. Download Udacity's Self-Driving Car Simulator (https://github.com/udacity/self-driving-car-sim). In my case it started normally on windows, but on linux it hung at startup.
2. Clone repository (git clone https://github.com/EvgeniiMaslov/self-driving-car-simulator.git)
3. Execute apply_model.py
4. Run Udacity's simulator
5. Select the map
6. Select autonomous mode


*(optional)* If you want to train model on your own data: 
1. Collect the data:
* run Udacity's simulator
* select the map
* select training mode
* push record button on the right
* select the folder of this repository
* push record button again
* drive 3+ circles in one direction, 3+ circles in opposite direction
* push pause button
2. Execute training.py


## How it works:

1. Simulator feeds an image to the script.
2. The image is pre-processed:
* cropped so that only part of the image with the road remains
* the color model changes to YUV, it makes the boundaries of the road more visible
* the image is resized and normalized
3. The CNN predicts the angle of rotation of the wheels from the image and sends it back to simulator
