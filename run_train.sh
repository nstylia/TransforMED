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

# PATH TO TRAINING DATA
export DATA_DIR=data/neoplasm/

# MAXIMUM SEQUENCE LENGTH
export MAXSEQLENGTH=128
export OUTPUTDIR=output/$TASK_NAME+$MAXSEQLENGTH/


# SELECT MODEL FOR FINE-TUNING

#export MODEL=bert-base-uncased
#export MODEL=monologg/biobert_v1.1_pubmed
export MODEL=monologg/scibert_scivocab_uncased
#export MODEL=roberta-base


python train.py \
  --model_type $MODELTYPE \
  --model_name_or_path $MODEL \
  --output_dir $OUTPUTDIR \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $DATA_DIR \
  --max_seq_length $MAXSEQLENGTH \
  --overwrite_output_dir \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --save_steps 1000 \
  --overwrite_cache #req for multiple choice