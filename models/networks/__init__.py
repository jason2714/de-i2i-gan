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
    save_path = opt.ckpt_dir / opt.name / load_fn
    weights = torch.load(save_path)
    net.load_state_dict(weights)
    return net.to(opt.device)
