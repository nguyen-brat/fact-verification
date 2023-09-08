#!/bin/bash
python model/sentence_retrieval/DPR/trainer.py  --train_data_path "train_data_path" \
                                                --max_length 256 \
                                                --epochs 100\
                                                --q_model "model name or path"\
                                                --ctx_encoder "model name or path"\
                                                --output_path "output path"

python model/reranking/trainer.py   --train_data_path "train_data_path" \
                                    --epochs 100\
                                    --model "model name"\
                                    --output_path "output path"

python model/claim_verification/gear/trainer.py --train_data_path "train_data_path" \
                                                --epochs 100\
                                                --model "model name"\
                                                --output_path "output path"