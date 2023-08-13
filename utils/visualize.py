import matplotlib.pyplot as plt
import matplotlib


def draw_mask_size_line_plot():
    # Use LaTeX-style font
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    mask_sizes = [1, 2, 4, 8, 16, 32, 64]
    fid_with_label = [77.7294, 75.2372, 72.5189, 66.1961, 65.8265, 68.8037, 68.7986]
    fid_without_label = [72.5320, 71.4595, 78.1524, 67.8896, 66.8889, 72.5428, 78.0656]
    fid_without_shifted_mask = [74.3047, 72.7787, 71.4838, 70.9607, 71.253, 77.5255, 74.8554]

    # Create the figure and the axes
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot FID scores for different mask sizes
    line1, = ax.plot(mask_sizes, fid_with_label, marker='o', linestyle='-', linewidth=1, markersize=5, label='Original')
    # line2, = ax.plot(mask_sizes, fid_without_label, marker='s', linestyle='-', linewidth=1, markersize=5,
    #                  label='Without Label')
    line3, = ax.plot(mask_sizes, fid_without_shifted_mask, marker='^', linestyle='-', linewidth=1, markersize=5,
                     label='Without Shifted Mask')

    # Emphasize the point at mask size 16
    ax.plot(16, fid_with_label[mask_sizes.index(16)], marker='o', markersize=7, color=line1.get_color())
    # ax.plot(16, fid_without_label[mask_sizes.index(16)], marker='s', markersize=7, color=line2.get_color())
    ax.plot(16, fid_without_shifted_mask[mask_sizes.index(16)], marker='^', markersize=7, color=line3.get_color())

    # Configure axes
    ax.set_xscale('log', base=2)  # to make x-axis log scale base 2
    ax.set_xticks(mask_sizes)  # to show all mask sizes on x-axis
    ax.set_xticklabels(mask_sizes, fontsize=10)  # set xtick labels with a suitable font size
    # for label in ax.get_xticklabels():
    #     if label.get_text() == '16':
    #         label.set_weight('bold')  # set the font weight of the label with mask size 16 to bold
    ax.set_yticklabels([f'{val:.2f}' for val in ax.get_yticks()],
                       fontsize=10)  # set ytick labels with a suitable font size

    # Set axis labels
    ax.set_xlabel('sask size', fontsize=12)
    ax.set_ylabel('FID', fontsize=12)

    # ax.set_title('FID Score for Different Mask Sizes', fontsize=16, pad=20)

    # # Add horizontal line at FID 70
    # ax.axhline(70, color='gray', linewidth=0.8, linestyle='dashed')

    # Set the legend
    ax.legend(fontsize=10)

    # Tight layout
    plt.tight_layout()

    plt.savefig('mask size and shifted mask.png', dpi=300)


def draw_mask_token_type_bar_plot():
    mask_token_types = ['zero', 'mean', 'scalar', 'vector', 'position', 'full']
    fid_scores = [68.9879, 75.2372, 74.8657, 74.6703, 65.8265, 67.02633]

    # Identify mask token types with FID lower than 70
    colors = ['0.3' if score < 70 else '0.8' for score in fid_scores]

    # Create the figure and the axes
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot FID scores for different mask token types, highlight the ones with FID lower than 70
    bars = ax.bar(mask_token_types, fid_scores, color=colors, edgecolor='black', width=0.5)  # Adjust width here

    # Set axis labels
    ax.set_xlabel('mask token type', fontsize=14)
    ax.set_ylabel('FID', fontsize=14)

    # Set y-axis limits to make differences in FID scores more visible
    ax.set_ylim([60, 80])

    # Adjust yticks and their labels
    ax.set_yticks(range(60, 81, 2))
    ax.set_yticklabels(range(60, 81, 2), fontsize=12)

    # Set xtick labels with a suitable font size
    ax.set_xticklabels(mask_token_types, fontsize=12)

    # Bold the 'position' label
    for label in ax.get_xticklabels():
        if label.get_text() == 'position':
            label.set_weight('bold')

    # Add horizontal line at FID 70
    ax.axhline(70, color='0.3', linewidth=0.8, linestyle='dashed')

    # Add bar value labels
    for bar in bars:
        yval = bar.get_height()
        ax.annotate(f'{yval:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, yval),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.savefig('mask token type.png', dpi=300)


def draw_mask_size_line_plot_with_value():
    # Use LaTeX-style font
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    mask_sizes = [1, 2, 4, 8, 16, 32, 64]
    fid_with_label = [77.7294, 75.2372, 72.5189, 66.1961, 65.8265, 68.8037, 68.7986]

    # Create the figure and the axes
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot FID scores for different mask sizes
    line1, = ax.plot(mask_sizes, fid_with_label, marker='o', linestyle='-', linewidth=1, markersize=5, label='Original MAE')

    # Add data point labels
    for i, txt in enumerate(fid_with_label):
        if i == 0:
            ax.annotate(round(txt, 2), (mask_sizes[i], fid_with_label[i]), xytext=(-10,-12), textcoords='offset points', fontsize=8)
        else:
            ax.annotate(round(txt, 2), (mask_sizes[i], fid_with_label[i]), xytext=(-5,8), textcoords='offset points', fontsize=8)

    # Emphasize the point at mask size 16
    ax.plot(16, fid_with_label[mask_sizes.index(16)], marker='o', markersize=7, color=line1.get_color())

    # Configure axes
    ax.set_xscale('log', base=2)  # to make x-axis log scale base 2
    ax.set_xticks(mask_sizes)  # to show all mask sizes on x-axis
    ax.set_xticklabels(mask_sizes, fontsize=10)  # set xtick labels with a suitable font size
    for label in ax.get_xticklabels():
        if label.get_text() == '16':
            label.set_weight('bold')  # set the font weight of the label with mask size 16 to bold
    ax.set_yticklabels([f'{val:.02f}' for val in ax.get_yticks()],
                       fontsize=10)  # set ytick labels with a suitable font size

    # Set axis labels
    ax.set_xlabel('mask size', fontsize=12)
    ax.set_ylabel('FID', fontsize=12)

    # ax.set_title('FID Score for Different Mask Sizes', fontsize=16, pad=20)

    # Set the legend
    ax.legend(fontsize=10)

    # Tight layout
    plt.tight_layout()

    plt.savefig('mask size.png', dpi=300)

def draw_mask_ratio_line_plot():
    # Use LaTeX-style font
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    mask_sizes = [10, 25, 40, 50, 60, 75, 90]
    fid_with_label = [73.3634, 72.6889, 64.9842, 66.8265, 67.3310, 65.8265, 80.5359]

    # Create the figure and the axes
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot FID scores for different mask sizes
    line1, = ax.plot(mask_sizes, fid_with_label, marker='o', linestyle='-', linewidth=1, markersize=5, label='Original MAE')

    # Add data point labels
    for i, txt in enumerate(fid_with_label):
        if i == len(fid_with_label) - 1:
            ax.annotate(round(txt, 2), (mask_sizes[i], fid_with_label[i]), xytext=(-10,-12), textcoords='offset points', fontsize=8)
        else:
            ax.annotate(round(txt, 2), (mask_sizes[i], fid_with_label[i]), xytext=(-5,8), textcoords='offset points', fontsize=8)

    # Emphasize the point at mask size 16
    ax.plot(75, fid_with_label[mask_sizes.index(75)], marker='o', markersize=7, color=line1.get_color())

    # Configure axes
    ax.set_xticks(mask_sizes)  # to show all mask sizes on x-axis
    ax.set_xticklabels(mask_sizes, fontsize=10)  # set xtick labels with a suitable font size
    for label in ax.get_xticklabels():
        if label.get_text() == '75':
            label.set_weight('bold')  # set the font weight of the label with mask size 16 to bold
    ax.set_yticklabels([f'{val:.02f}' for val in ax.get_yticks()],
                       fontsize=10)  # set ytick labels with a suitable font size

    # Set axis labels
    ax.set_xlabel('mask ratio (%)', fontsize=12)
    ax.set_ylabel('FID', fontsize=12)

    # ax.set_title('FID Score for Different Mask Sizes', fontsize=16, pad=20)

    # Set the legend
    ax.legend(fontsize=10)

    # Tight layout
    plt.tight_layout()

    plt.savefig('mask ratio.png', dpi=300)
    