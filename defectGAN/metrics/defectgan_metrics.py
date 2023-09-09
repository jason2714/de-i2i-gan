from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm
import numpy as np
import torch
from metrics.fid_score import calculate_fid, calculate_fid_from_model
from itertools import combinations


@torch.no_grad()
def calculate_metrics_from_model(opt, model, bg_loader, df_loader, metric_models, metrics, is_score_type='min_max'):
    if 'fid' in metrics:
        calculate_fid_and_is_from_model(opt, model, bg_loader, df_loader, metric_models, metrics, is_score_type)
    if 'lpips' in metrics:
        calculate_lpips_from_model(opt, model, bg_loader, df_loader, metric_models, metrics)
    if 'mfid' in metrics:
        calculate_mfid_from_model(opt, model, bg_loader, df_loader, metric_models, metrics)
    return metrics


def calculate_lpips_from_model(opt, model, bg_loader, df_loader, metric_models, metrics):
    lpips_scores = []
    image_nums = 0
    label_iters = iter(df_loader)
    batch_size = len(next(label_iters)[1])
    num_df_imgs = len(df_loader.dataset)
    pbar = tqdm(range(0, num_df_imgs + batch_size, batch_size), colour='MAGENTA')
    for _ in pbar:
        pbar.set_description('Calculating LPIPS...')
        bg_data, _, _ = next(bg_loader)
        _, df_labels, _ = next(label_iters)
        bg_data, df_labels = bg_data.to(device=opt.device), df_labels.to(device=opt.device)
        bg_data = bg_data[:df_labels.size(0)]
        for slice_bg_img, df_label in zip(bg_data, df_labels):
            batch_bg_img = slice_bg_img.unsqueeze(0).expand(opt.num_lpips_images, -1, -1, -1)
            batch_df_label = df_label.unsqueeze(0).expand(opt.num_lpips_images, -1)
            fake_imgs, _ = model('inference', batch_bg_img, batch_df_label)
            comb_imgs = torch.stack([torch.stack(img_pair) for img_pair in combinations(fake_imgs.clamp(-1, 1), 2)])
            lpips_scores.append(metric_models['lpips'](comb_imgs[:, 0], comb_imgs[:, 1]))
        image_nums += df_labels.size(0)
    metrics['lpips'] = torch.mean(torch.stack(lpips_scores)).item()
    return metrics


def calculate_fid_and_is_from_model(opt, model, bg_loader, df_loader, metric_models, metrics, is_score_type='min_max'):
    if 'fid' in metrics:
        assert opt.npz_path is not None, 'npz_path should not be None if calculate FID score'
    logits_arr = []
    pred_arr = []
    pbar = tqdm(df_loader, colour='MAGENTA')
    # BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE
    for df_data, df_labels, _ in pbar:
        pbar.set_description(f'Calculating metrics of IS and FID...')
        bg_data, bg_labels, _ = next(bg_loader)
        bg_data, bg_labels = bg_data[:df_labels.size(0)], bg_labels[:df_labels.size(0)]
        fake_imgs, _ = model('inference', bg_data, df_labels)

        if 'fid' or 'is' in metrics:
            fake_imgs = (fake_imgs + 1) / 2
            logits, pred = metric_models['inception'](fake_imgs)
            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if logits.size(2) != 1 or logits.size(3) != 1:
                logits = adaptive_avg_pool2d(logits, output_size=(1, 1))
            logits_arr.append(logits.view(logits.size(0), -1).cpu())
            pred_arr.append(pred.view(pred.size(0), -1).cpu())
    if 'fid' in metrics:
        logits_arr = torch.cat(logits_arr, dim=0).numpy()
        mu = np.mean(logits_arr, axis=0)
        sigma = np.cov(logits_arr, rowvar=False)
        fid_value = calculate_fid((opt.npz_path, [mu, sigma]), opt.batch_size, opt.device, opt.dims,
                                  num_workers=4, model=metric_models['inception'])
        metrics['fid'] = fid_value
    if 'is' in metrics:
        splits = 10
        pred_arr = torch.cat(pred_arr, dim=0)
        # calculate probs and logits
        prob = pred_arr.softmax(dim=1)
        log_prob = pred_arr.log_softmax(dim=1)

        # split into groups
        prob = prob.chunk(splits, dim=0)
        log_prob = log_prob.chunk(splits, dim=0)

        # calculate score per split
        mean_prob = [p.mean(dim=0, keepdim=True) for p in prob]
        kl_ = [p * (log_p - m_p.log()) for p, log_p, m_p in zip(prob, log_prob, mean_prob)]
        kl_ = [k.sum(dim=1).mean().exp() for k in kl_]
        kl = torch.stack(kl_)

        # return mean and std
        mu, sigma = kl.mean().item(), kl.std().item()
        if is_score_type == 'min_max':
            max_score = mu + 2 * sigma
            min_score = mu - 2 * sigma
            metrics['is'] = {'mean': mu,
                             'max': max_score,
                             'min': min_score}
        elif is_score_type == 'std':
            metrics['is'] = {'mean': mu,
                             'std': sigma}
    return metrics


def calculate_mfid_from_model(opt, model, bg_loader, df_loader, metric_models, metrics):
    # load stats for each class from npz file
    assert opt.npy_path is not None, 'npy_path should not be None'
    class_stats = np.load(opt.npy_path, allow_pickle=True).item()

    class_fids = []
    pbar = tqdm(range(1, opt.label_nc), colour='MAGENTA')
    # BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE
    for class_idx in pbar:
        pbar.set_description('Calculating metrics of mFID...')
        label = [0] * opt.label_nc
        label[class_idx] = 1
        label = tuple(label)
        fid_value = calculate_fid_from_model(opt, model, metric_models['inception'],
                                             bg_loader, label,
                                             class_stats[label])
        class_fids.append(fid_value)
    print('FID for each class:', class_fids)
    metrics['mfid'] = sum(class_fids) / len(class_fids)
    return metrics
