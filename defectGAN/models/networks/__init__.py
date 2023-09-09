import torch


def save_network(net, net_label, epoch, opt):
    save_fn = f'{epoch}_net_{net_label}.pth'
    save_dir = opt.ckpt_dir / opt.name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / save_fn
    torch.save(net.state_dict(), save_path)
    # if len(opt.gpu_ids) and torch.cuda.is_available():
    #     net.cuda()


def load_network(net, net_label, epoch, opt):
    load_fn = f'{epoch}_net_{net_label}.pth'
    save_path = opt.ckpt_dir / opt.load_model_name / load_fn
    weights = torch.load(save_path)
    # TODO load weights from different model state_dict
    # weights = {k.replace('spade_', '').replace('sean_', ''): v for k, v in weights.items()}
    # # TODO ignore mlp_latent!!!!!
    weights = {k.replace('spade_', '').replace('sean_', ''): v for k, v in weights.items() if 'mlp_latent' not in k}
    net.load_state_dict(weights, strict=False)
    return net.to(opt.device, non_blocking=True)
