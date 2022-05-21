==== train by TT demo
python main.py --Rate_lm 0.1 --Rate_lpts 0 --batch_size 16
python main.py --checkpoint your_checkpoint_name.pth --start_epoch 242 --lr 0.00025
python main.py --train_by_TTT --Rate_lpts 0 --Rate_lm 0.1 --batch_size 16 --checkpoint your_checkpoint_name.pth --start_epoch 242 --lr 0.00025

==== train by data demo
python main.py --train_by_TT False --model_name HAN  --checkpoint HAN_x4_L1__Set5_psnr_32.007_ssim_0.892.pth  --Rate_lm 0.1  --Rate_lpts 0 --batch_size 16
python main.py --train_by_TT False --model_name HAN  --checkpoint HAN_x4_L1_0.1Lm.pth  --Rate_lm 0.2 --Rate_lpts 0 --batch_size 16
python main.py --train_by_TT False --model_name RFDN_L  --Rate_lm 0.05 --Rate_lpts 0 --batch_size 16

==== test demo
python main.py --Train False --model_name xxx --checkpoint xxx.pth
python main.py --Train False --model_name RFDN --checkpoint RFDN_x4_t1_IMDN_t2_EdgeSRN_L1_0.1Lpts.pth