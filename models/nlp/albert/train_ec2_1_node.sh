
cd /shared/deep-learning-models/models/nlp/
export PYTHONPATH=${PYTHONPATH}:${PWD}

herringsinglenode \
bash /shared/deep-learning-models/models/nlp/albert/launch.sh \
python /shared/deep-learning-models/models/nlp/albert/run_pretraining.py \
--train_dir=/shared/data/albert/tfrecords/train/max_seq_len_512_max_predictions_per_seq_20_masked_lm_prob_15 \
--val_dir=/shared/data/albert/tfrecords/validation/max_seq_len_512_max_predictions_per_seq_20_masked_lm_prob_15 \
--log_dir=/shared/logs \
--checkpoint_dir=/shared/checkpoints \
--load_from=scratch \
--model_type=albert \
--model_size=base \
--per_gpu_batch_size=32 \
--gradient_accumulation_steps=16 \
--warmup_steps=3125 \
--total_steps=500 \
--learning_rate=0.00176 \
--optimizer=lamb \
--log_frequency=10

