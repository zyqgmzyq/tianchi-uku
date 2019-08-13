import argparse

parser = argparse.ArgumentParser(description='main')
# parser.add_argument('--model', default='RCAN')
# parser.add_argument('--model', default='WDSR')
parser.add_argument('--model', default='RCAN')
parser.add_argument('--scale', type=int, default=4)  # upscaling factor
parser.add_argument('--repeat', type=int, default=2)  # times for iterating total trainset during an epoch
parser.add_argument('--patch_size', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--repoch', '-re', type=int, default=0)  # recover from ckpt
parser.add_argument('--pre_train', default='True')  # using pretrained ckpt
parser.add_argument('--lr', type=float, default=1e-4)

parser.add_argument('--mark', default='')  # additional info
parser.add_argument('--loss', default='yuv', help='l1, yuv, char, l2')  # additional info
parser.add_argument('--milestones', '-ms', default='850-1500-25000', help='such as 200-500-1000-1500-2000-3000')

parser.add_argument('--debug', action='store_true')  # less data for fast debugging
parser.add_argument('--dN', type=int, default=5000, help='val data num')  # num of val sample
parser.add_argument('--dS', type=int, default=100, help='val data skip stride')  # stride fo val sample
parser.add_argument('--dR', type=int, default=10, help='train data skip stride')  # stride fo train sample
parser.add_argument('--load_mem', default='none', help='lr, all...')  # pre-loading specified data into memory

parser.add_argument('--sd', type=float, default=255.)
parser.add_argument('--mean', type=float, default=0.5)
parser.add_argument('--seed', default=666)

parser.add_argument('--ensemble', action='store_true')
parser.add_argument('--ckpt', default='./checkpoint/s4_rcan_ps32_bs16_lossYUV.ckpt')
parser.add_argument('--is_psnr', action='store_true')

parser.add_argument('--train_path', default='./data/round2_train/train_np', help='path to save your train set')
parser.add_argument('--val_path', default='./data/round2_val/val_np', help='path to save your validation set')
parser.add_argument('--infer_path', default='./data/round2_testb/test_np', help='path to save your test set')
parser.add_argument('--output_path', default='./data/submit', help='path to save your generated bmp file')

# CNN param
parser.add_argument('--n_resblocks', type=int, default=20)
parser.add_argument('--n_feats', type=int, default=128)
parser.add_argument('--begin_id', type=int, default=900, help='begin_id, same as actual id')
parser.add_argument('--mid_id', type=int, default=905, help='5 more than begin_id')
parser.add_argument('--end_id', type=int, default=950, help='50 more than begin_id')
parser.add_argument('--zip_dir', default='../data/round2_train_input', help='path to save testc')

args = parser.parse_args()

message = '{0}parameters{0}\n'.format('='*9)

for arg in vars(args):  # 把命令行可能的字符串布尔变成真的布尔，不弄这个偶尔会出错
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

    message += '{}: {}\n'.format(arg, vars(args)[arg])

args.message = message
print(message)

