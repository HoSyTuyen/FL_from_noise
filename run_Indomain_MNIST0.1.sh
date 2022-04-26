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
               --KD_weight 1 \
               --KD_lr 0.0005 \
               --KD_bs 2048 \
               --KD_latent_size 100 \
               --KD_num_per_client 2048 \
               --pretrained_G "None" \
               --writer "real_images_track_KL" \
               --train_with_img True \
               --track_KL True
