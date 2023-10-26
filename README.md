

# PRNU baseline evaluation on DivNoise dataset
The code is based on [this work](https://github.com/polimi-ispl/prnu-python).

## Usage 🔑

### Dataset 📁
The *DivNoise* dataset can be downloaded at [https://divnoise.fotoverifier.eu/](https://divnoise.fotoverifier.eu/). Part 1 refers to smartphones, tablets and webcams, while the remaining parts contain the Canon cameras. 

When using the *DivNoise* dataset, in order to comply with the code, the downloaded dataset must be slightly modified by splitting the natural images of each camera into separate `Train` and `Test` sets (within the last `JPG` subfolder). Afterwards, simply include the dataset in the `test` folder (`test/Dataset/`) and everything should work.

In case you use your own dataset:
- Firstly, the dataset has to be included in the `test` folder (`test/Dataset/`).
- Within the `Dataset`, each camera's images must be contained in its respective folder, named as `Brand_Model_CameraLocation_ID`, where camera location is either *Frontal* or *Rear*. In particular, they should be divided into separate `Train` and `Test` splits, as explained above.
- The images belonging to a specific camera should have the name in the form *Brand_Model_CamLocation_ID_Content_X.jpg*, where 'Content' identifies the image type (*flat* or *natural*) and 'X' is an incremental number (example: *Apple_iPadmini5_Frontal_0_Nat_0.jpg*)

### Compute fingerprints of the cameras 📷
The fingerprint of each camera is computed from the set of 50 corresponding flat images. Just run: 
```
python3 compute_fingerprints.py
```
The fingerprints of the cameras are saved in the `Fingerprints` directory. You may need to first create that folder.
### Test on images 📊
The noise residuals of query images are then computed from 100 test images for each camera, and compared with the camera fingerprints through PCE. Each test image is assigned to the camera that results in the highest PCE (among the candidate cameras).
```
python3 test_images.py
```
The charts showing the performance of the method are saved in the `plots` folder. You may need to first create that folder.
### Test on videos 🎞️
Both frame-level and video-level attribution scenarios are evaluated to benchmark the basic PRNU scheme. Frames can be extracted form videos using either [OpenCV](https://pypi.org/project/opencv-python/) or [FFmpeg](https://www.ffmpeg.org/).
#### Frame-level test
At frame-level, each frame is treated independently, as if they were individual unrelated images.
```
python3 test_videos.py
```
#### Video-level test
At video-level, camera fingerprints are derived from the training videos and matched against test videos on a per-frame basis using PCE. The source camera is determined by majority voting across all frame-level matches.
```
python3 video_level_test.py
```
Tested with Python >= 3.4

## Credits
Reference MATLAB implementation by Binghamton university: 
http://dde.binghamton.edu/download/camera_fingerprint/

