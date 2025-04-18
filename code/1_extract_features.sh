#!/bin/bash

# Data directory
DATASET_DIR='/egr/research-deeptech/elelukeh/MOS_project/MusicEval-baseline/data/MusicEval-phase1'

# Feature directory
FEATURE_DIR='/egr/research-deeptech/elelukeh/MOS_project/MusicEval-baseline/data/features/'

# Workspace
WORKSPACE='/egr/research-deeptech/elelukeh/MOS_project/MusicEval-baseline/code'
cd $WORKSPACE

# feature types: 'logmel'
FEATURE_TYPE='logmel'

# Extract Features
python utils/feature_extractor.py --dataset_dir=$DATASET_DIR --feature_dir=$FEATURE_DIR --feature_type=$FEATURE_TYPE 
