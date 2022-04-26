python main.py --dataset Mnist-alpha0.1-ratio0.5 \
               --algorithm FedSP \
               --batch_size 32 \
               --num_glob_iters 500 \
               --local_epochs 20 \
               --num_users 10 \
               --lamda 1 \
               --learning_rate 0.01 \
               --model cnn \
               --personal_learning_rate 0.01 \
               --times 1 \
               --KD_epoch 20 \
               --KD_weight 40 \
               --KD_temperature 4 \
               --KD_lr 0.04 \
               --KD_bs 4096 \
               --KD_latent_size 100 \
               --KD_num_per_client 4096 \
               --pretrained_G "FLAlgorithms/trainmodel/pretrained/G_emnist_200E.ckpt" \
               --writer "G_EMNIST_MNIST0.1_trackKL_2048" \
               --KD_selective_feature_ratio 1.0 \
               --track_KL True 