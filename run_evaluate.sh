#!/usr/bin/env bash

# ACTIVATE VIRTUAL ENVIRONMENT
source activate test

# SELECT TASK

# 1) SEQUENCE TAGGING
export TASK_NAME=seqtag
export MODELTYPE=bert-seqtag

# 2) RELATION CLASSIFICATION
#export TASK_NAME=relclass
#export MODELTYPE=bert

# PATH TO TEST DATA
export DATA_DIR=data/neoplasm/
#export DATA_DIR=data/glaucoma_test/
#export DATA_DIR=data/mixed_test/

# MAXIMUM SEQUENCE LENGTH
export MAXSEQLENGTH=128


# EVALUATE MODEL:
export MODEL=output/seqtag128SciBERT/
export OUTPUTDIR=$MODEL



python train.py \
  --model_type $MODELTYPE \
  --model_name_or_path $MODEL \
  --output_dir $OUTPUTDIR \
  --task_name $TASK_NAME \
  --do_eval \
  --do_lower_case \
  --data_dir $DATA_DIR \
  --max_seq_length $MAXSEQLENGTH \
  --overwrite_output_dir \
  --overwrite_cache \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 1.0 \
  --save_steps 1000