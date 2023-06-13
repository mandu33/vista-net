#work1 command
python train_old.py --hidden_dim 50 --att_dim 100 --num_images 3 --batch_size 32 --learning_rate 0.001 --num_epochs 20
#work2 command
python train.py --hidden_dim 50 --att_dim 100 --num_images 3 --batch_size 32 --learning_rate 0.001 --num_epochs 20 --use_lstm true

#pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
#tensor-flow
conda install tensorflow==1.13.1


#作业1
python test.py --hidden_dim 50 --att_dim 100 --num_images 3 --batch_size 32 --learning_rate 0.001 --num_epochs 20
nohup python test.py --hidden_dim 50 --att_dim 100 --num_images 3 --batch_size 32 --learning_rate 0.001 --num_epochs 20
#作业2
python test_lstm.py --hidden_dim 50 --att_dim 100 --num_images 3 --batch_size 32 --learning_rate 0.001 --num_epochs 10 --use_lstm true
#作业3
python resnet18.py