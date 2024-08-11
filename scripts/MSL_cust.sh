export CUDA_VISIBLE_DEVICES=0

python3 main.py --anomaly_ratio 1 --num_epochs 2  --batch_size 128  --mode train --dataset MSL --data_path dataset/MSL \
                --input_c 55 --output_c 55 --d_model 128 --e_layers 3 --n_heads 8
python3 main.py --anomaly_ratio 1 --num_epochs 2 --batch_size 128 --mode test --dataset MSL --data_path dataset/MSL \
                --input_c 55 --output_c 55 --d_model 128 --e_layers 3 --n_heads 8




