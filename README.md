

# PRNU baseline evaluation on DivNoise dataset
The code is based on [this work](https://github.com/polimi-ispl/prnu-python).

## Usage
- Firstly, the dataset has to be included in the `test` folder.
- Within the `Dataset`, each camera's images must be contained in its respective folder, named as `Brand_Model_CameraLocation_ID`, where camera location is either *Frontal* or *Rear*. In particular, they should be divided into separate `Train` and `Test` splits.
- The images belonging to a specific camera should have the name in the form *Brand_Model_CamLocation_ID_Content_X.jpg*, where 'Content' identifies the image type (*flat* or *natural*) and 'X' is an incremental number (example: *Apple_iPadmini5_Frontal_0_Nat_0.jpg*)

### Compute fingerprints of the cameras
```
python3 compute_fingerprints.py
```
### Test on images
```
python3 test_images.py
```

### Test on video frames
```
python3 test_videos.py
```
### Test at video level
```
python3 video_level_test.py
```
The fingerprints of the cameras are saved in the `Fingerprints` directory, while the charts showing the performance of the method are saved in the `plots` folder.
Tested with Python >= 3.4

## Credits
Reference MATLAB implementation by Binghamton university: 
http://dde.binghamton.edu/download/camera_fingerprint/

