# Dog pain detection algorihtm

We use two stream ConvLSTM and LSTM model to detect dogs' pain action from the RGB video frames and corresponding body keypoints. 
## Data
All the code detail is in the ./dataset 
### Video frames --> croped images
* We extract video frames from our own OFT dog pain video dataset. Applying pretrained YOLOv5 algorithm to get bounding box of dogs, and crop the image frame according to the bbox, make sure the dogs' body take a majority of area in the image. All the images are re-scaled to (224X224).
### Croped images --> body keypoints
* We apply pretrained HeNet algorithm to extract 17 keypoints within the bbox
* Applying our own missing data treatment algorihtm to filter and complement the keypoints graph
## Model
All the code detail is in the ./lib
### ConvLSTM
* ConvLSTM is used to extract spatial information from the RGB video frame. Input data of the ConvLSTM is a video clip consist of a stack of RGN frames. The input shape is (N, 224, 224), N is length of video clip
### LSTM 
* LSTM with attention mechenism is applied to process the keypoints graph. Input data of the LSTM is a stack of keypoints 2-D coordinates. The input shape is (N, 2X17), N is length of video clip, is the number of keypoints.
## Train strategies
All code detail can be found in ./tool/train
## Install

```
```

## Usage

```
```

## Contributing

PRs accepted.

## License

MIT © Richard McRichface
