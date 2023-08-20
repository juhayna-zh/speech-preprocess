from .BSRNN import BSRNN

def create_BSRNN(opt):
    opt = opt['model']
    model = BSRNN(
        sr=opt['sr'],
        win=opt['win'],
        stride=opt['stride'],
        feature_dim=opt['feature_dim'],
        num_layer=opt['num_layer'],
        num_spk_layer=opt['num_spk_layer'],
        same_mask=opt['same_mask']
    )
    return model
