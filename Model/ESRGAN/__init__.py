import torch

def get_ESRGAN(checkpoint):
    from .RRDBNet_arch import RRDBNet
    model = RRDBNet(3, 3, 64, 23, gc=32)
    if checkpoint is not None:
        checkpoint = './Checkpoints/ESRGAN/' + checkpoint
        model.load_state_dict(torch.load(checkpoint), strict=True)

    return model