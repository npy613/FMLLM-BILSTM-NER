#!/usr/bin/env python3

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

def load_attention_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def crop_valid_tokens(tokens, labels, attn_matrix, pad_token="[PAD]"):
   
    valid_idx = [i for i, t in enumerate(tokens) if t != pad_token]
    tokens = [tokens[i] for i in valid_idx]
    labels = [labels[i] for i in valid_idx]
    attn_matrix = attn_matrix[np.ix_(valid_idx, valid_idx)]
    return tokens, labels, attn_matrix


def crop_submatrix(tokens, labels, attn_matrix, start=None, end=None):
   
    if start is None or end is None:
        return tokens, labels, attn_matrix

    tokens = tokens[start:end]
    labels = labels[start:end]
    attn_matrix = attn_matrix[start:end, start:end]
    return tokens, labels, attn_matrix


def build_label_color_map(pred_labels):
    entity_types = sorted(set(lbl.split('-', 1)[1]
                              for lbl in pred_labels if lbl != 'O'))

    paper_colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488',
                    '#F39B7F', '#8491B4', '#91D1C2', '#DC0000']

    type2color = {et: paper_colors[i % len(paper_colors)]
                  for i, et in enumerate(entity_types)}

    def lbl_color(lbl):
        return type2color[lbl.split('-', 1)[1]] if lbl != 'O' else '#555555'

    return type2color, lbl_color



def get_entity_spans(labels):
   
    spans = []
    start = None
    prev_entity = None

    for i, lbl in enumerate(labels):
        if lbl != 'O':
            entity_type = lbl.split('-', 1)[1] if '-' in lbl else lbl
            if start is None:
                start = i
                prev_entity = entity_type
            elif entity_type != prev_entity:
               
                spans.append((start, i - 1, prev_entity))
                start = i
                prev_entity = entity_type
        else:
            if start is not None:
                spans.append((start, i - 1, prev_entity))
                start = None
                prev_entity = None

    
    if start is not None:
        spans.append((start, len(labels) - 1, prev_entity))

    return spans


def draw_heatmap(ax, matrix, tokens, labels, lbl_color,
                 title, cmap='Blues', vmin=None, vmax=None,
                 diff_mode=False, diff_matrix=None, improvement_threshold=0.01):

    im = ax.imshow(matrix, cmap=cmap, aspect='auto',
                   vmin=vmin, vmax=vmax)

    n = len(tokens)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tokens, rotation=90, fontsize=7)
    ax.set_yticklabels(tokens, fontsize=7)

    for i, (xtl, ytl) in enumerate(zip(ax.get_xticklabels(),
                                       ax.get_yticklabels())):
        c = lbl_color(labels[i])
        if labels[i] != 'O':
            xtl.set_fontweight('bold')
            ytl.set_fontweight('bold')
        xtl.set_color(c)
        ytl.set_color(c)

    ax.set_xlabel('Key Tokens', fontsize=8)
    ax.set_ylabel('Query Tokens', fontsize=8)
    ax.set_title(title, fontsize=9)

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.3)
    ax.tick_params(which='minor', bottom=False, left=False)

   
    if diff_mode and diff_matrix is not None:
        entity_spans = get_entity_spans(labels)
        
        frame_color = '#2C3E50'  
       
        for i, (start1, end1, _) in enumerate(entity_spans):
            for j, (start2, end2, _) in enumerate(entity_spans):
               
                if i != j:
                    continue

                region_diff = diff_matrix[start1:end1+1, start2:end2+1]
                mean_improvement = region_diff.mean()
                
                if mean_improvement > improvement_threshold:
                    
                    x_min = start2 - 0.5
                    x_max = end2 + 0.5
                    y_min = start1 - 0.5
                    y_max = end1 + 0.5

                    rect = mpatches.Rectangle(
                        (x_min, y_min),
                        x_max - x_min,
                        y_max - y_min,
                        linewidth=1.8,
                        edgecolor=frame_color,
                        facecolor='none',
                        linestyle='--',
                        alpha=0.9
                    )
                    ax.add_patch(rect)

    return im

def plot_comparison_heatmaps(
        json_path1,
        json_path2,
        output_pdf="attention_comparison.pdf",
        crop_pad=True,
        sub_start=None,
        sub_end=None,
        diff_percentile_clip=99,
        save_individual=False
):

    data1 = load_attention_data(json_path1)
    data2 = load_attention_data(json_path2)

    tokens = data1['tokens']
    labels1 = data1['pred_labels']
    labels2 = data2['pred_labels']

    A = np.array(data1['attention_matrix'])
    B = np.array(data2['attention_matrix'])

    if crop_pad:
        tokens, labels1, A = crop_valid_tokens(tokens, labels1, A)
        _, labels2, B = crop_valid_tokens(tokens, labels2, B)

    tokens, labels1, A = crop_submatrix(tokens, labels1, A,
                                       sub_start, sub_end)
    _, labels2, B = crop_submatrix(tokens, labels2, B,
                             sub_start, sub_end)

    diff = B - A

    global_max = 0.21 
    print("Global max attention:", global_max)

    clip_val = np.percentile(np.abs(diff), diff_percentile_clip)

    
    type2color1, lbl_color1 = build_label_color_map(labels1)
    type2color2, lbl_color2 = build_label_color_map(labels2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im1 = draw_heatmap(axes[0], A, tokens, labels1,
                       lbl_color1, "Model A",
                       cmap='Blues',
                       vmin=0, vmax=global_max)

    im2 = draw_heatmap(axes[1], B, tokens, labels2,
                       lbl_color2, "Model B",
                       cmap='Blues',
                       vmin=0, vmax=global_max)

    im3 = draw_heatmap(axes[2], diff, tokens, labels2,
                       lbl_color2, "Δ Attention (B − A)",
                       cmap='RdBu_r',
                       vmin=-clip_val,
                       vmax=clip_val,
                       diff_mode=True,
                       diff_matrix=diff,
                       improvement_threshold=0.0)

    fig.colorbar(im1, ax=axes[0], fraction=0.046)
    fig.colorbar(im2, ax=axes[1], fraction=0.046)
    fig.colorbar(im3, ax=axes[2], fraction=0.046)

    merged_type2color = {**type2color1, **type2color2}
    legend_handles = [mpatches.Patch(color='#555555', label='O')]
    for et, c in merged_type2color.items():
        legend_handles.append(mpatches.Patch(color=c, label=et))

    fig.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05),
        ncol=min(len(legend_handles), 6),
        frameon=True,
        fontsize=8,
        title="NER Labels"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])  

    with PdfPages(output_pdf) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')

    plt.close()
    print("Saved to:", output_pdf)
    print("Mean |Δ|:", np.abs(diff).mean())

    if save_individual:
        base = output_pdf.replace('.pdf', '')
        configs = [
            (A, tokens, labels1, lbl_color1, "Model A", 'Blues', 0, global_max, False, None, f"{base}_modelA.pdf"),
            (B, tokens, labels2, lbl_color2, "Model B", 'Blues', 0, global_max, False, None, f"{base}_modelB.pdf"),
            (diff, tokens, labels2, lbl_color2, "Δ Attention (B − A)", 'RdBu_r', -clip_val, clip_val, True, diff, f"{base}_diff.pdf"),
        ]
        merged_type2color = {**type2color1, **type2color2}
        legend_handles = [mpatches.Patch(color='#555555', label='O')]
        for et, c in merged_type2color.items():
            legend_handles.append(mpatches.Patch(color=c, label=et))

        for matrix, toks, lbls, lbl_fn, title, cmap, vmin, vmax, diff_mode, diff_mat, path in configs:
            fig_s, ax_s = plt.subplots(1, 1, figsize=(7, 6))
            im = draw_heatmap(ax_s, matrix, toks, lbls, lbl_fn, title,
                              cmap=cmap, vmin=vmin, vmax=vmax,
                              diff_mode=diff_mode, diff_matrix=diff_mat,
                              improvement_threshold=0.0)
            fig_s.colorbar(im, ax=ax_s, fraction=0.046)
            fig_s.legend(handles=legend_handles, loc='upper center',
                         bbox_to_anchor=(0.5, 1.05),
                         ncol=min(len(legend_handles), 6),
                         frameon=True, fontsize=8, title="NER Labels")
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            with PdfPages(path) as pdf:
                pdf.savefig(fig_s, dpi=300, bbox_inches='tight')
            plt.close()
            print("Saved to:", path)




if __name__ == "__main__":

    json_path1 = "/src/ablation-llama/LLaMA-Bi-LSTM/attention_data_0.json"
    json_path2 = "/src/FmLLaMA-Bi-LSTM/attention_data_1.json"

    plot_comparison_heatmaps(
        json_path1,
        json_path2,
        output_pdf="attention_comparison.pdf",
        crop_pad=True,
        sub_start=None,   # 可设置为实体区域起点
        sub_end=None,     # 可设置为实体区域终点
        diff_percentile_clip=99,
        save_individual=True  # 同时保存三张独立 PDF
    )

