from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm
import numpy as np
import torch
from metrics.fid_score import calculate_fid


@torch.no_grad()
def calculate_metrics_from_model(opt, model, bg_loader, df_loader, metric_models, metrics):
    if 'fid' in metrics:
        assert opt.npz_path is not None, 'npz_path should not be None if calculate FID score'
    logits_arr = []
    pred_arr = []
    lpips_scores = []
    pbar = tqdm(df_loader, colour='MAGENTA')
    # BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE
    for df_data, df_labels, _ in pbar:
        pbar.set_description(f'Calculating metrics of {metrics.keys()}...')
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
        if 'lpips' in metrics:
            lpips_score = metric_models['lpips'](fake_imgs, df_data.to(device=opt.device))
            lpips_scores.append(lpips_score)
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
        max_score = mu + 2 * sigma
        min_score = mu - 2 * sigma
        metrics['is'] = {'mean': mu,
                         'max': max_score,
                         'min': min_score}

    if 'lpips' in metrics:
        lpips = sum(lpips_scores) / len(lpips_scores)
        metrics['lpips'] = lpips
    return metrics
