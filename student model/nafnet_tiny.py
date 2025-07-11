# student_training/nafnet_tiny.py

import torch

from basicsr.models.archs.NAFNet_arch import NAFNet  # ✅ corrected import

def get_nafnet_tiny():
    return NAFNet(
        img_channel=3,
        width=16,
        middle_blk_num=2,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2]
    )
