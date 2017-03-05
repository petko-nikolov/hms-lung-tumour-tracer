#!/bin/bash
TEST_EXAMPLES_DIR=$1
MODELS_DIR=$2
PREDICTIONS_FILE=$3
WORK_DIR=workdir
source $HOME/hms/bin/activate

python convert_to_dataset.py --input_dir $TEST_EXAMPLES_DIR --output_file $WORK_DIR/test_annotations.txt   
python create_records.py --input_dir $TRAIN_EXAMPLES_DIR --input_annotations $WORK_DIR/test_annotations.txt \
  --output_directory $WORK_DIR/test_data

python train.py --data_dir $WORK_DIR/test_data --model_dir $WORK_DIR/models/radiomics_vgg--batch_size 4 \ 
  --checkpoint $WORK_DIR/models/lungs/model-30000 --validate_output_dir $WORK_DIR/predictions/test_lungs

python slice_lung.py --input_dir $TEST_EXAMPLES_DIR --predictions_dir $WORK_DIR/predictions/test_lungs \
  --output_directory $WORK_DIR/lung_test_examples

python create_records.py --input_dir $TEST_EXAMPLES_DIR --input_annotations $WORK_DIR/test_annotations.txt \
  --output_directory $WORK_DIR/test_data_radiomics --lungs

python train.py --data_dir $WORK_DIR/test_data_radiomics --model_dir $WORK_DIR/models/radiomics_vgg --batch_size 12 \
  --model_variation radiomics_vgg --checkpoint $WORK_DIR/models/radiomics_vgg/model-16000 \
  --validate_output_dir $WORK_DIR/predictions/test_radiomics 

python generate_contours.py --predictions_dir $WORK_DIR/predictions/test_radiomics --input_annotations $WORK_DIR/test_annotations.txt \
  --input_lungs_dir $WORK_DIR/lung_test_examples --input_dir $TEST_EXAMPLES_DIR --output_contours_path $PREDICTIONS_FILE 

