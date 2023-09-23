

# PRNU baseline evaluation on DivNoise

The code is based on [this work](https://github.com/polimi-ispl/prnu-python), whose authors are:
- Luca Bondi (luca.bondi@polimi.it)
- Paolo Bestagini (paolo.bestagini@polimi.it)
- Nicolò Bonettini (nicolo.bonettini@polimi.it)

Test implementation by:
- Alberto Casagrande (alberto.casagrande@studenti.unitn.it) - University of Trento

## Usage
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

Tested with Python >= 3.4

## Credits
Reference MATLAB implementation by Binghamton university: 
http://dde.binghamton.edu/download/camera_fingerprint/

