import random
import numpy as np
import torch
import logging
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def use_gpu(gpu_ids=0):
    use_cuda = torch.cuda.is_available()
    # print(use_cuda)
    device = torch.device(f'cuda:{gpu_ids}' if use_cuda else 'cpu')
    print('Device used:', device)
    if use_cuda:
        torch.cuda.set_device(device)
    return device


def fix_rand_seed(seed=123):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def init_logging(name, level=logging.INFO):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=level,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


def generate_mask(image_size, patch_size, mask_ratio):
    """
        input image_size: (b, c, h, w)
        output size: (b, 1, h, w)
    """
    height_size = image_size[2] // patch_size
    width_size = image_size[3] // patch_size
    mask = torch.bernoulli(torch.full((image_size[0], 1, height_size, width_size), (1 - mask_ratio)))

    return F.interpolate(mask, scale_factor=patch_size, mode='nearest')


def generate_shifted_mask(image_size, patch_size, mask_ratio):
    """
        input image_size: (b, c, h, w)
        output size: (b, 1, h, w)
    """
    img_h, img_w = image_size[2], image_size[3]
    h_shift = torch.randint(low=0, high=patch_size, size=(1,))
    w_shift = torch.randint(low=0, high=patch_size, size=(1,))
    extend_image_size = (*image_size[:2], img_h + patch_size, img_w + patch_size)
    extend_mask = generate_mask(extend_image_size, patch_size, mask_ratio)
    mask = extend_mask[:, :, h_shift: h_shift + img_h, w_shift: w_shift + img_w]
    return mask


def calc_mean_std(feat, eps=1e-5):
    """
        input feat: (b, c, h, w) or (b, n, c)
        return mean: (b, c, 1, 1), std: (b, c, 1, 1)
    """
    if feat.dim() == 3:
        return calc_embed_mean_std(feat, eps=eps)
    elif feat.dim() == 4:
        return calc_feat_mean_std(feat, eps=eps)
    else:
        raise ValueError('Wrong feature dimension: {}'.format(feat.dim()))


def calc_feat_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def calc_embed_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 3)
    N, _, C = size[:3]
    feat_var = feat.var(dim=1) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.mean(dim=1).view(N, C, 1, 1)
    return feat_mean, feat_std


def calc_kl_with_logits(p, q, temperature=4.):
    """
        temperature: float, make the distribution less sharp
        input type of q and q should be in log scale
    """
    kl_mean = F.kl_div(
        F.log_softmax(q / temperature, dim=1),
        F.log_softmax(p / temperature, dim=1),
        reduction='batchmean', log_target=True
    ) * temperature * temperature
    return kl_mean


def visualize_embeddings(embeddings, plt_dir, plt_name, reduction_type='pca'):
    plt_dir.mkdir(parents=True, exist_ok=True)
    plt_path = plt_dir / plt_name

    rand_state = 0
    all_embeddings = torch.cat(list(torch.stack(embedding_tensors) for embedding_tensors in embeddings.values()))
    embedding_labels = [label for label, embedding_tensors in embeddings.items() for _ in embedding_tensors]
    label_strs = dict()
    for label in embeddings.keys():
        label_str = [str(idx) for idx, label_value in enumerate(label) if label_value == 1]
        label_strs[label] = '-'.join(label_str)
    embedding_strings = list(map(lambda x: label_strs[x], embedding_labels))
    all_embeddings = all_embeddings.numpy()

    if reduction_type == 'pca':
        pca = PCA(n_components=50, random_state=rand_state)
        all_embeddings = pca.fit_transform(all_embeddings)
    elif reduction_type == 'tsne':
        tsne = TSNE(n_components=2, random_state=rand_state, n_jobs=-1)
        all_embeddings = tsne.fit_transform(all_embeddings)
    else:
        raise NotImplementedError

    color_map = {label: plt.cm.tab20(idx) for idx, label in enumerate(embeddings.keys())}
    embedding_colors = [color_map[label] for label in embedding_labels]

    plt.figure(figsize=(12, 12))

    x_min, x_max = all_embeddings.min(0), all_embeddings.max(0)
    X_norm = (all_embeddings - x_min) / (x_max - x_min)  # Normalize
    for idx in range(len(all_embeddings)):
        plt.text(X_norm[idx, 0], X_norm[idx, 1], embedding_strings[idx], fontsize=6, color=embedding_colors[idx])
    plt.xticks([])
    plt.yticks([])
    plt.savefig(plt_path)


def calc_embeddings_mean_variance(embeddings):
    label_strs = dict()
    for label in embeddings.keys():
        label_str = [str(idx) for idx, label_value in enumerate(label) if label_value == 1]
        label_strs[label] = '-'.join(label_str)
    for label, embedding_lists in embeddings.items():
        embedding_tensors = torch.stack(embedding_lists)
        mean = embedding_tensors.mean(dim=0)
        var = embedding_tensors.var(dim=0)
        embeddings[label] = [mean, var]
    for first_label in embeddings.keys():
        for second_label in embeddings.keys():
            if first_label != second_label:
                mean1, var1 = embeddings[first_label]
                mean2, var2 = embeddings[second_label]
                print(f'{label_strs[first_label]:^8} vs {label_strs[second_label]:^8}: '
                      f'dist={torch.dist(mean1, mean2):.2f}, var1={var1.mean():.2f}, var2={var2.mean():.2f}')


def label_to_str(label):
    label_str = [str(idx) for idx, label_value in enumerate(label) if label_value == 1]
    return '-'.join(label_str)


def generate_multilabel_combinations(label_dim):
    binary_values = torch.tensor([0, 1])
    all_combinations = torch.cartesian_prod(*([binary_values] * label_dim))
    return all_combinations
