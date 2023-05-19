from tqdm import tqdm


def calculate_lpips_from_model(opt, model, lpips, test_loaders):
    pbar = tqdm(test_loaders['defects'], colour='MAGENTA')
    # BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE
    lpips_scores = []
    for df_data, df_labels, _ in pbar:
        pbar.set_description('Calculating lpips score...')
        bg_data, bg_labels, _ = next(test_loaders['background'])
        bg_data, bg_labels = bg_data[:df_labels.size(0)], bg_labels[:df_labels.size(0)]
        fake_imgs, _ = model('inference', bg_data, df_labels)
        lpips_score = lpips(fake_imgs, df_data.to(device=opt.device))
        lpips_scores.append(lpips_score)
    return sum(lpips_scores) / len(lpips_scores)


def main():
    opt, model, test_loaders = None, None, None
    cal_lpips_score(opt, model, test_loaders)


if __name__ == '__main__':
    main()
