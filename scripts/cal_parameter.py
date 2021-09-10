import torch


def cal(state_dict):
    num = 0
    for k, v in state_dict.items():
        shape = v.shape
        t = 1
        for i in shape:
            t = t*i
        num = num+t
    p = '{:.1f}M'.format(num/1000000)
    return num, p


if __name__ == '__main__':
    targets = {
        'Liif': '../response_code/liif/save/cyto/epoch-2.pth',
        'VDSR': '../log/checkpoints/SR-Refocus/VDSR/netG_epoch_2_40.pth',
        'SRResNet': '../log/checkpoints/SR-Refocus/SRResNet/netG_epoch_2_59.pth',
        'CARN': '../log/checkpoints/SR-Refocus/CARN/netG_epoch_2_60.pth',
        'EDSR': '../log/checkpoints/SRSOTA/EDSR_0.25_512_depth/netG_epoch_2_40.pth',
        'SRCNN': '../log/checkpoints/SR-Refocus/SRCNN/netG_epoch_2_60.pth',
        'MWCNN': '../log/checkpoints/SR-Refocus/MWCNN/netG_epoch_2_60.pth',
        'Deep-Z': '../log/checkpoints/SR-Refocus/Deep-Z-SR/netG_epoch_2_39.pth',
        'RCAN': '../log/checkpoints/SR-Refocus/RCAN/netG_epoch_2_60.pth',
        'RFANet': '../log/checkpoints/response/RFANet_v1/G_99_7873.pth',
        'STSRNet': '../log/checkpoints/our/STTRNet2-TransformerV5-plus-4/G_59_4713.pth',
        'CrossNet': '../response_code/ECCV2018_CrossNet_RefSR/checkpoints/CP100.pth',
        'SRNTT': '../response_code/srntt-pytorch/runs/ref_hr_8blocks/netG_091.pth',
        'TTSR': '../response_code/TTSR-master/train/Our/TTSR_lrref/model/model_00100.pt',
    }
    for k, v in targets.items():
        #print(k)
        state_dict =torch.load(v)
        num1, num2 = cal(state_dict)
        print('{}:{},{}'.format(k, num1, num2))
