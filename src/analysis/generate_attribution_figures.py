import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

__all__ = [
    "load_all_attributions",
    "find_best_record",
    "create_phobert_attribution_figure",
    "create_method_agreement_figure",
    "create_cross_model_agreement_figure",
]

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'DejaVu Sans',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

OUTPUT_DIR = "paper/figures"


def load_all_attributions(attr_dir="results/attributions"):
    records = []
    for f in sorted(os.listdir(attr_dir)):
        if f.endswith('.pkl'):
            with open(os.path.join(attr_dir, f), 'rb') as fp:
                records.append(pickle.load(fp))
    return records


def find_best_record(records):
    def score(r):
        return (len(r.get('attributions', {})), len(r.get('text', '')))
    return max(records, key=score) if records else None

def create_phobert_attribution_figure(record, save_path):
    attributions = record.get('attributions', {})

    phobert_methods = {}
    method_labels = {
        'phobert_shap': 'PhoBERT — SHAP',
        'phobert_ig': 'PhoBERT — Integrated Gradients',
        'phobert_rollout': 'PhoBERT — Attention Rollout',
    }

    for key, label in method_labels.items():
        if key in attributions:
            phobert_methods[label] = attributions[key]

    if not phobert_methods:
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.set_axis_off()
        ax.text(0.5, 0.5, 'PhoBERT Attribution\n(No data available)',
                ha='center', va='center', fontsize=16, transform=ax.transAxes)
        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return

    n = len(phobert_methods)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), squeeze=False)

    for row_idx, (method_name, attrs) in enumerate(phobert_methods.items()):
        ax = axes[row_idx, 0]
        tokens = attrs.get('tokens', [])
        scores = np.asarray(attrs.get('scores', []))

        if len(tokens) == 0:
            ax.text(0.5, 0.5, '(No tokens)', ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            continue

        abs_max = float(np.max(np.abs(scores))) if np.any(scores != 0) else 1.0

        cell_h = 1.0
        for i, (tok, sc) in enumerate(zip(tokens, scores)):
            if abs_max > 0:
                norm_score = sc / abs_max
            else:
                norm_score = 0

            if sc >= 0:
                color = plt.cm.Reds(0.3 + 0.7 * abs(norm_score))
            else:
                color = plt.cm.Blues(0.3 + 0.7 * abs(norm_score))

            alpha = 0.3 + 0.7 * abs(norm_score) if abs_max > 0 else 0.5
            text_color = 'white' if alpha > 0.55 else 'black'

            ax.add_patch(plt.Rectangle(
                (i, 0), 1.0, cell_h,
                facecolor=color, alpha=alpha,
                edgecolor='white', linewidth=0.8,
            ))

            display_tok = str(tok).replace('@@', '').replace('▁', '')
            if len(display_tok) > 8:
                display_tok = display_tok[:7] + '…'
            ax.text(
                i + 0.5, cell_h / 2, display_tok,
                ha='center', va='center',
                fontsize=10,
                color=text_color,
                fontweight='medium',
            )

        ax.set_xlim(0, len(tokens))
        ax.set_ylim(0, cell_h)
        ax.set_title(
            method_name,
            fontsize=14, fontweight='bold', loc='left', pad=10,
            color='#333333'
        )
        ax.set_axis_off()

    fig.suptitle(
        f'PhoBERT Token Attribution (Example #{record.get("id", "?")})',
        fontsize=16, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_method_agreement_figure(records, save_path):
    all_methods = set()
    for r in records:
        for k in r.get('attributions', {}):
            all_methods.add(k)

    method_labels = {
        'lr_shap': 'LR — SHAP',
        'svm_shap': 'SVM — SHAP',
        'bilstm_ig': 'BiLSTM — IG',
        'bilstm_grad': 'BiLSTM — Grad',
        'phobert_shap': 'PhoBERT — SHAP',
        'phobert_ig': 'PhoBERT — IG',
        'phobert_rollout': 'PhoBERT — Rollout',
    }

    available = sorted([m for m in all_methods if m in method_labels])
    available_labels = [method_labels.get(m, m) for m in available]
    n = len(available)

    if n < 2:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_axis_off()
        ax.text(0.5, 0.5, 'Method Agreement\n(Not enough data)',
                ha='center', va='center', fontsize=16, transform=ax.transAxes)
        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return

    matrix = np.eye(n, dtype=np.float64)
    counts = np.zeros((n, n), dtype=np.int64)

    for r in records:
        aligned = {}
        for m in available:
            if m in r.get('attributions', {}):
                scores = np.asarray(r['attributions'][m]['scores'])
                if len(scores) > 0:
                    aligned[m] = scores

        if len(aligned) < 2:
            continue

        for i, mi in enumerate(available):
            if mi not in aligned:
                continue
            for j, mj in enumerate(available):
                if mj not in aligned:
                    continue
                if len(aligned[mi]) != len(aligned[mj]):
                    continue

                scores_i = aligned[mi]
                scores_j = aligned[mj]

                top_k = min(10, len(scores_i))
                if top_k <= 0:
                    continue

                top_i = set(np.argsort(np.abs(scores_i))[-top_k:])
                top_j = set(np.argsort(np.abs(scores_j))[-top_k:])
                jaccard = len(top_i & top_j) / len(top_i | top_j) if top_i | top_j else 0

                matrix[i, j] = matrix[i, j] * counts[i, j] + jaccard
                counts[i, j] += 1

    for i in range(n):
        for j in range(n):
            if counts[i, j] > 0:
                matrix[i, j] /= counts[i, j]

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * n + 2), max(6, 0.9 * n + 1)))

    cmap = plt.cm.viridis

    im = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect='equal')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(available_labels, rotation=35, ha='right', fontsize=12)
    ax.set_yticklabels(available_labels, fontsize=12)

    ax.set_title(
        'Mean Rank-Agreement (Jaccard Top-10)\nAcross All Test Examples',
        fontsize=14, fontweight='bold', pad=20
    )

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if counts[i, j] == 0:
                color = 'gray'
                text = '—'
            elif val < 0.5:
                color = 'white'
                text = f'{val:.2f}'
            else:
                color = 'black'
                text = f'{val:.2f}'
            ax.text(j, i, text, ha='center', va='center',
                    color=color, fontsize=12, fontweight='medium')

    cbar_ax = fig.add_axes([0.88, 0.35, 0.03, 0.5])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Jaccard Similarity', fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_cross_model_agreement_figure(record, save_path):
    attributions = record.get('attributions', {})
    text = record.get('text', '')
    words = str(text).split()

    if len(words) == 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_axis_off()
        ax.text(0.5, 0.5, 'Cross-Model Agreement\n(No text data)',
                ha='center', va='center', fontsize=16, transform=ax.transAxes)
        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return

    model_colors = {
        'lr_shap': '#2E86AB',      # Blue
        'svm_shap': '#F18F01',     # Orange
        'bilstm_ig': '#2CA02C',    # Green
        'phobert_rollout': '#C73E1D',  # Red
    }

    model_names = {
        'lr_shap': 'Logistic Regression',
        'svm_shap': 'SVM',
        'bilstm_ig': 'BiLSTM',
        'phobert_rollout': 'PhoBERT',
    }

    available_models = [m for m in ['lr_shap', 'svm_shap', 'bilstm_ig', 'phobert_rollout']
                       if m in attributions]

    if len(available_models) < 2:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_axis_off()
        ax.text(0.5, 0.5, 'Cross-Model Agreement\n(At least 2 models needed)',
                ha='center', va='center', fontsize=16, transform=ax.transAxes)
        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return

    word_scores = {m: {} for m in available_models}

    for model_name in available_models:
        attrs = attributions[model_name]
        tokens = attrs.get('tokens', [])
        scores = np.asarray(attrs.get('scores', []))

        if len(tokens) != len(scores):
            continue

        abs_max = np.max(np.abs(scores)) if np.any(scores) else 1.0
        if abs_max > 0:
            scores = scores / abs_max

        for tok, sc in zip(tokens, scores):
            tok_clean = str(tok).replace('@@', '').replace('▁', '').replace('_', ' ')
            for w in words:
                w_clean = w.replace('_', ' ').lower()
                if tok_clean.lower() == w_clean or tok_clean.lower() in w_clean:
                    word_scores[model_name][w] = word_scores[model_name].get(w, 0) + abs(sc)
                    break

    all_word_scores = {}
    for model_name in available_models:
        for w, s in word_scores[model_name].items():
            all_word_scores[w] = all_word_scores.get(w, 0) + abs(s)

    top_words = sorted(all_word_scores.keys(),
                      key=lambda w: all_word_scores[w],
                      reverse=True)[:15]

    if not top_words:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_axis_off()
        ax.text(0.5, 0.5, 'Cross-Model Agreement\n(No comparable tokens)',
                ha='center', va='center', fontsize=16, transform=ax.transAxes)
        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return

    n_models = len(available_models)
    n_words = len(top_words)
    fig_height = max(6, 0.4 * n_words + 2)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    bar_width = 0.8 / n_models
    positions = np.arange(n_words)

    for i, model_name in enumerate(available_models):
        scores = [word_scores[model_name].get(w, 0) for w in top_words]
        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.barh(
            positions + offset, scores,
            height=bar_width * 0.9,
            label=model_names.get(model_name, model_name),
            color=model_colors.get(model_name, f'C{i}'),
            alpha=0.85,
            edgecolor='white',
            linewidth=0.5
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(top_words, fontsize=11)
    ax.set_xlabel('Normalized Attribution Score', fontsize=13)
    ax.set_title(
        'Cross-Model Token Attribution Comparison\n(Top 15 Most Important Words)',
        fontsize=14, fontweight='bold', pad=15
    )
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved: {save_path}")


def main():
    print("=" * 60)
    print("GENERATING IMPROVED PAPER FIGURES")
    print("=" * 60)

    attr_dir = "results/attributions"
    if not os.path.exists(attr_dir):
        print(f" Attribution directory not found: {attr_dir}")
        return

    records = load_all_attributions(attr_dir)
    print(f" Loaded {len(records)} attribution records")

    if not records:
        print(" No attribution data found")
        return

    best_record = find_best_record(records)
    print(f" Using showcase example #{best_record.get('id', '?')}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n Generating figures...")

    create_phobert_attribution_figure(
        best_record,
        os.path.join(OUTPUT_DIR, 'fig_token_attribution_phobert.png')
    )

    create_method_agreement_figure(
        records,
        os.path.join(OUTPUT_DIR, 'fig_method_agreement.png')
    )

    create_cross_model_agreement_figure(
        best_record,
        os.path.join(OUTPUT_DIR, 'fig_cross_model_agreement.png')
    )

    print("\n" + "=" * 60)
    print("ALL FIGURES GENERATED!")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
