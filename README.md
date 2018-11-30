# Lung Tumour Tracer

My solution for the Harvard Medical School Lung Tumour Tracer Competition organized by TopCoder. It consists of two stacked deep networks. The first network
takes as an input 3 images of consecutive slices from a patient CT Scan and outputs a mask of the lungs. The second network segments the tumorous
pixels from the cropped lung regions. 
The two networks architecture is a UNet model with a VGG16 backbone pretrained on ImageNet.

### Installation

```
./compile.sh
```

### Training
```
./train.sh $DATA_DIR $MODEL_OUTPUT_DIR
```

### Testing 

```
./test.sh $TEST_DATA_DIR $MODEL_OUTPUT_DIR/model.ckpt-N
```
