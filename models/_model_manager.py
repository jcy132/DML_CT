from models.cutGAN import cutGAN_model

def create_model(opt):
    if opt.model == 'cutGAN':
        return cutGAN_model(opt)
    else:
        raise NotImplementedError('Model is not defined')