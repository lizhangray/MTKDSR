# -*- coding: utf-8 -*-
# @Time    : 8/16/2021 3:14 PM
# @Author  : YaoGengqi
# @FileName: main.py
# @Software: PyCharm
# @Description: Training the teacher

from Trainer import Trainer
import argparse

parser = argparse.ArgumentParser(description="Initial the trainer setting.")

# the model would to be trained or tested.
parser.add_argument("--model_name", type=str, default="EdgeSPSR")
parser.add_argument("--checkpoint", type=str, default=None)

# teacher_1 model with high score of psnr and ssim.
# teacher_2 model with high score of PI or else.
# if you want to train by the dataset instead of the teacher's result,
# you should set the --train_by_TT False in your shell command.
parser.add_argument("--train_by_TT",             default=False)
parser.add_argument("--train_by_TTT",            default=False,     action="store_true")
parser.add_argument("--t1_model_name", type=str, default="IMDN")
parser.add_argument("--t1_checkpoint", type=str, default="IMDN_x4.pth")
parser.add_argument("--t2_model_name", type=str, default="ESRGAN", help='lpts')
parser.add_argument("--t2_checkpoint", type=str, default="ESRGAN_x4.pth")

# Train setting
# the argument --Train means you want to train the model instead of test the model.
# you can change the rate of loss in your shell command to train your model.
parser.add_argument("--Train",                  default=True)
parser.add_argument("--Rate_l1",    type=float, default=1,          help="L1损失的超参数，用于和PSNR值高的结果作比较")
parser.add_argument("--Rate_lpts",  type=float, default=0.1,          help="Lpts损失超参数，用于和AG等值高的结果作比较")
parser.add_argument("--Rate_lm",    type=float, default=0,       help="Lm损失的超参数，用于和PI等值高的结果作比较")


# you can choose the valid_dataset like set5 or set14 or else.
parser.add_argument("--valid_dataset", type=str, default="Set14", help="Set5/14, BSD100, Urban109...")

# the start epoch and the end epoch.
# if you change the start epoch, remember to change your lr too.
parser.add_argument("--start_epoch",    type=int, default=1)
parser.add_argument("--nEpochs",        type=int, default=800)

# adjust the lr and the parameters of the SGD method.
parser.add_argument("--lr",             type=float, default=1e-3,   help="Initial Learn rate")
parser.add_argument("--lr_step_size",   type=int,   default=50,     help="SGD options")
parser.add_argument("--lr_gamma",       type=float, default=9e-1)

# the setting of the train_dataset.
parser.add_argument("--batch_size",     type=int, default=16)
parser.add_argument("--num_workers",    type=int, default=8, help="output patch size")
parser.add_argument("--n_train",        type=int, default=800, help="number of training set")

# Test Setting
# you should change the argument --Train False in your shell command.
parser.add_argument("--test_datasets", type=list,   default=['Set5', 'Set14','BSD100', 'Urban100','Manga109'])
parser.add_argument("--if_save_image",              default=True)
parser.add_argument("--output_folder", type=str,    default='./Result')

# Use default is enough
parser.add_argument("--seed",   type=int, default=77)
parser.add_argument("--root",   type=str, default="../torch_SR/data",   help='dataset directory')
parser.add_argument("--scale",  type=int, default=4,                    help="super-resolution scale")
parser.add_argument("--phase",  type=str, default='train')

parser.add_argument("--test_every", type=int, default=1000)
parser.add_argument("--n_val",      type=int, default=1, help="number of validation set")

# image setting
parser.add_argument("--isY",                    default=True,   action="store_true")
parser.add_argument("--ext",        type=str,   default='.png', help='png or npy')
parser.add_argument("--n_colors",   type=int,   default=3,      help="number of color channels to use")
parser.add_argument("--rgb_range",  type=int,   default=1,      help="maxium value of RGB")
parser.add_argument("--patch_size", type=int,   default=192,    help="output patch size")

args = parser.parse_args()

def display_args(args):

    print("#######################################################################################################")

    print("===> Model_name         : " + str(args.model_name))
    print("===> Model_checkpoint   : " + str(args.checkpoint))

    if args.Train == True:
        print("===> Learn_rate_init    : " + str(args.lr))
        print("===> Rate_L1_loss       : " + str(args.Rate_l1))
        print("===> Rate_Lm_loss       : " + str(args.Rate_lm))
        print("===> Rate_Lpts_loss     : " + str(args.Rate_lpts))
        print("===> Batch_size         : " + str(args.batch_size))

        if args.train_by_TT == True:
            print("===> T1_Model_name      : " + str(args.t1_model_name))
            print("===> T1_Model_checkpoint: " + str(args.t1_checkpoint))
            print("===> T2_Model_name      : " + str(args.t2_model_name))
            print("===> T2_Model_checkpoint: " + str(args.t2_checkpoint))

    else:
        print("===> Test_datasets: " + str(args.test_datasets))

    print("#######################################################################################################")

if __name__ == '__main__':

    trainer = Trainer(args)
    if args.Train == True:
        display_args(args)
        trainer.Train()
    else:
        trainer.Test()

