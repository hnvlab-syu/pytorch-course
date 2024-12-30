import torch

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.archs.edsr_arch import EDSR


def create_model(model: str, weight_path: str | tuple, upscale: int):   # pretrained_path: args.weights
    '''
    if model.lower() == 'esrgan':
        model = RRDBNet(
            num_in_ch=3,  # RGB
            num_out_ch=3, # RGB
        )
        if weight_path:  # 'weights/ESRGAN.pth'
            weights = torch.load(weight_path, weights_only=True)    # .pth 파일 안의 가중치 데이터만 로드, 실행 가능한 객체 무시 (unpickle의 잠재적인 보안 문제를 예방)
            if 'params' in weights:
                weights = weights['params']
            model.load_state_dict(weights, strict=False)

    elif model.lower() == 'realesrgan':
        model = RRDBNet(
            num_in_ch=12,   # 3->12
            num_out_ch=3,
            scale=upscale,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
        )
        if weight_path:
            weights = torch.load(weight_path, weights_only=True)
            if 'params' in weights:
                weights = weights['params']
            model.load_state_dict(weights, strict=False)
    '''

    if model.lower() == 'edsr':
        model = EDSR(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=16,
                upscale=upscale,
                res_scale=1,
                img_range=255.,
                rgb_mean=(0.4488, 0.4371, 0.4040)
            )
        if weight_path:
            weights = torch.load(weight_path, weights_only=True)
            if 'params' in weights:
                weights = weights['params']
            model.load_state_dict(weights, strict=False)

    for param in model.parameters():
        param.requires_grad = True

    return model