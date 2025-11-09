import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def visualise_mlp_flatten():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    colors = {
        'input': '#E8F4F8',
        'feature': '#B8E6F0',
        'policy': '#FF6B6B',
        'value': '#4ECDC4',
        'output': '#FFD700',
        'activation': '#E6E6FA'
    }
    
    def draw_box(x, y, w, h, text, color, fontsize=10, bold=False):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                            edgecolor='black', facecolor=color, linewidth=2)
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
               fontsize=fontsize, weight=weight, wrap=True)
    
    def draw_arrow(x1, y1, x2, y2, color='black', width=2):
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->',
                               color=color, linewidth=width, mutation_scale=20)
        ax.add_patch(arrow)
    
    # title
    ax.text(5, 9.5, 'Maskable PPO with FlattenExtractor (Baseline MLP)', 
           ha='center', fontsize=16, weight='bold')
    
    # input observation
    draw_box(4.5, 8, 1, 0.6, 'Observation\n(1002)', colors['input'], 11, True)
    
    # feature extractors (flatten)
    draw_box(2, 6.8, 1.5, 0.6, 'Policy Feature\nExtractor\n(Flatten)', colors['feature'], 9)
    draw_box(6.5, 6.8, 1.5, 0.6, 'Value Feature\nExtractor\n(Flatten)', colors['feature'], 9)
    
    # arrows from input to feature extractors
    draw_arrow(4.7, 8, 2.75, 7.4, 'gray')
    draw_arrow(5.3, 8, 7.25, 7.4, 'gray')
    
    # policy network on left
    draw_box(1.5, 5.5, 1.2, 0.5, 'Linear\n1002→256', colors['policy'], 9)
    draw_box(1.5, 4.8, 1.2, 0.3, 'Tanh', colors['activation'], 8)
    draw_box(1.5, 4, 1.2, 0.5, 'Linear\n256→128', colors['policy'], 9)
    draw_box(1.5, 3.3, 1.2, 0.3, 'Tanh', colors['activation'], 8)
    draw_box(1.3, 2, 1.6, 0.6, 'Action Net\nLinear 128→290', colors['output'], 10, True)
    
    # policy network arrows
    draw_arrow(2.75, 6.8, 2.1, 6, '#FF69B4')
    draw_arrow(2.1, 5.5, 2.1, 5.1, '#FF69B4')
    draw_arrow(2.1, 4.8, 2.1, 4.5, '#FF69B4')
    draw_arrow(2.1, 4, 2.1, 3.6, '#FF69B4')
    draw_arrow(2.1, 3.3, 2.1, 2.6, '#FF69B4')
    
    # value network on right
    draw_box(7.3, 5.5, 1.2, 0.5, 'Linear\n1002→256', colors['value'], 9)
    draw_box(7.3, 4.8, 1.2, 0.3, 'Tanh', colors['activation'], 8)
    draw_box(7.3, 4, 1.2, 0.5, 'Linear\n256→128', colors['value'], 9)
    draw_box(7.3, 3.3, 1.2, 0.3, 'Tanh', colors['activation'], 8)
    draw_box(7.1, 2, 1.6, 0.6, 'Value Net\nLinear 128→1', colors['output'], 10, True)
    
    # value network arrows
    draw_arrow(7.25, 6.8, 7.9, 6, '#228B22')
    draw_arrow(7.9, 5.5, 7.9, 5.1, '#228B22')
    draw_arrow(7.9, 4.8, 7.9, 4.5, '#228B22')
    draw_arrow(7.9, 4, 7.9, 3.6, '#228B22')
    draw_arrow(7.9, 3.3, 7.9, 2.6, '#228B22')
    
    # network labels
    ax.text(2.1, 6.5, 'Policy Network', ha='center', fontsize=11, 
           weight='bold', color='#C71585')
    ax.text(7.9, 6.5, 'Value Network', ha='center', fontsize=11, 
           weight='bold', color='#006400')
    
    # output labels
    ax.text(2.1, 1.5, 'Action Logits\n(290 actions)', ha='center', fontsize=9, style='italic')
    ax.text(7.9, 1.5, 'State Value\n(scalar)', ha='center', fontsize=9, style='italic')
    
    info_text = "Simple MLP baseline\nFlattened observation\n→ Policy/Value networks"
    draw_box(0.2, 0.2, 2, 0.8, info_text, '#F0F0F0', 8)
    
    plt.tight_layout()
    return fig


def visualise_combined_extractor():
    fig, ax = plt.subplots(figsize=(20, 14))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    colors = {
        'input': '#E8F4F8',
        'cnn': '#FFB6C1',
        'mlp': '#B0E0B0',
        'combine': '#DDA0DD',
        'policy': '#FF6B6B',
        'value': '#4ECDC4',
        'output': '#FFD700',
        'activation': '#E6E6FA',
        'pooling': '#FFE4B5',
        'norm': '#FFF8DC'
    }
    
    def draw_box(x, y, w, h, text, color, fontsize=8, bold=False):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", 
                            edgecolor='black', facecolor=color, linewidth=1.5)
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
               fontsize=fontsize, weight=weight, wrap=True)
    
    def draw_arrow(x1, y1, x2, y2, color='black', style='->', width=2):
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                               color=color, linewidth=width, mutation_scale=15)
        ax.add_patch(arrow)
    
    # title
    ax.text(7, 13.2, 'Maskable PPO with COMBINED Feature Extractor', 
           ha='center', fontsize=18, weight='bold')
    ax.text(7, 12.7, '(Dual-path: CNN for board + MLP for numeric features)', 
           ha='center', fontsize=10, style='italic', color='gray')
    
    # input observations
    draw_box(1.5, 11.5, 1.5, 0.6, 'Board\nObservation\n(20 channels)', colors['input'], 9, True)
    draw_box(11, 11.5, 1.5, 0.6, 'Numeric\nObservation\n(76 features)', colors['input'], 9, True)
    
    # cnn path on left
    y_pos = 10.5
    x_left = 0.8
    
    # convolutional block 1
    draw_box(x_left, y_pos, 1.1, 0.35, 'Conv2d\n20→64, 3x3', colors['cnn'], 7)
    draw_box(x_left+1.2, y_pos, 0.9, 0.35, 'BN+ReLU', colors['activation'], 7)
    y_pos -= 0.45
    draw_box(x_left, y_pos, 1.1, 0.35, 'Conv2d\n64→64, 3x3', colors['cnn'], 7)
    draw_box(x_left+1.2, y_pos, 0.9, 0.35, 'BN+ReLU', colors['activation'], 7)
    y_pos -= 0.45
    draw_box(x_left+0.5, y_pos, 1.1, 0.3, 'MaxPool2d\n2×2, s=2', colors['pooling'], 7)
    
    # conv block 2
    y_pos -= 0.45
    draw_box(x_left, y_pos, 1.1, 0.35, 'Conv2d\n64→128, 3x3', colors['cnn'], 7)
    draw_box(x_left+1.2, y_pos, 0.9, 0.35, 'BN+ReLU', colors['activation'], 7)
    y_pos -= 0.45
    draw_box(x_left, y_pos, 1.1, 0.35, 'Conv2d\n128→128, 3x3', colors['cnn'], 7)
    draw_box(x_left+1.2, y_pos, 0.9, 0.35, 'BN+ReLU', colors['activation'], 7)
    y_pos -= 0.45
    draw_box(x_left+0.5, y_pos, 1.1, 0.3, 'MaxPool2d\n2×2, s=2', colors['pooling'], 7)
    
    # 3
    y_pos -= 0.45
    draw_box(x_left, y_pos, 1.1, 0.35, 'Conv2d\n128→256, 3x3', colors['cnn'], 7)
    draw_box(x_left+1.2, y_pos, 0.9, 0.35, 'BN+ReLU', colors['activation'], 7)
    y_pos -= 0.45
    draw_box(x_left, y_pos, 1.1, 0.35, 'Conv2d\n256→256, 3x3', colors['cnn'], 7)
    draw_box(x_left+1.2, y_pos, 0.9, 0.35, 'BN+ReLU', colors['activation'], 7)
    y_pos -= 0.45
    draw_box(x_left, y_pos, 1.1, 0.35, 'Conv2d\n256→256, 3x3', colors['cnn'], 7)
    draw_box(x_left+1.2, y_pos, 0.9, 0.35, 'BN+ReLU', colors['activation'], 7)
    
    y_pos -= 0.45
    draw_box(x_left+0.5, y_pos, 1.1, 0.3, 'AdaptiveAvg\nPool 3×2', colors['pooling'], 7)
    y_pos -= 0.4
    draw_box(x_left+0.5, y_pos, 1.1, 0.3, 'Flatten\n→ 1536', colors['cnn'], 7)
    
    # fully connected layers
    y_pos -= 0.5
    draw_box(x_left+0.1, y_pos, 1.9, 0.4, 'Linear 1536→256\nLayerNorm, ReLU, Dropout', colors['cnn'], 8)
    
    ax.text(1.9, 11, 'CNN Branch', ha='center', fontsize=11, 
           weight='bold', color='#C71585')
    
    # mlp on right
    y_pos = 10.5
    x_right = 10.5
    
    draw_box(x_right, y_pos, 2, 0.45, 'Linear 76→256\nLayerNorm, ReLU, Dropout', colors['mlp'], 8)
    y_pos -= 0.6
    draw_box(x_right, y_pos, 2, 0.45, 'Linear 256→256\nLayerNorm, ReLU, Dropout', colors['mlp'], 8)
    y_pos -= 0.6
    draw_box(x_right, y_pos, 2, 0.45, 'Linear 256→256\nLayerNorm, ReLU', colors['mlp'], 8)
    
    ax.text(11.5, 11.1, 'MLP Branch', ha='center', fontsize=11, 
           weight='bold', color='#006400')
    
    # combined layers
    draw_box(5.5, 4.5, 3, 0.6, 'Concatenate Features\n[CNN: 256 + MLP: 256] = 512', 
             colors['combine'], 10, True)
    
    #fc layers
    draw_box(5.5, 3.6, 3, 0.5, 'Linear 512→512\nLayerNorm, ReLU, Dropout 0.1', colors['combine'], 9)
    draw_box(5.5, 2.9, 3, 0.5, 'Linear 512→512\nLayerNorm, ReLU', colors['combine'], 9)
    
    #arrows
    draw_arrow(2, 5.3, 7, 5.1, '#FF69B4', '->', 2.5)
    draw_arrow(11.5, 9, 7, 5.1, '#228B22', '->', 2.5)
    
    #feature extractor out
    draw_box(5.5, 2.1, 3, 0.4, 'Feature Vector (512)', colors['combine'], 9, True)
    
    # pi and vf networks
    draw_box(3, 1.2, 2.5, 0.4, 'Policy MLP\nLinear 512→256, Tanh', colors['policy'], 8)
    draw_box(3, 0.7, 2.5, 0.4, 'Linear 256→128, Tanh', colors['policy'], 8)
    draw_box(3, 0.2, 2.5, 0.4, 'Action Net\nLinear 128→290', colors['output'], 9, True)
    
    draw_box(8.5, 1.2, 2.5, 0.4, 'Value MLP\nLinear 512→256, Tanh', colors['value'], 8)
    draw_box(8.5, 0.7, 2.5, 0.4, 'Linear 256→128, Tanh', colors['value'], 8)
    draw_box(8.5, 0.2, 2.5, 0.4, 'Value Net\nLinear 128→1', colors['output'], 9, True)
    
    # arrows
    draw_arrow(7, 2.1, 4.25, 1.6, '#FF6B6B', '->', 2.5)
    draw_arrow(7, 2.1, 9.75, 1.6, '#4ECDC4', '->', 2.5)
    
    # labels
    ax.text(4.25, -0.1, 'Action Logits (290)', ha='center', fontsize=9, style='italic')
    ax.text(9.75, -0.1, 'State Value', ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    return fig


def visualise_board_only():
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    colors = {
        'input': '#E8F4F8',
        'cnn': '#FFB6C1',
        'policy': '#FF6B6B',
        'value': '#4ECDC4',
        'output': '#FFD700',
        'activation': '#E6E6FA',
        'pooling': '#FFE4B5'
    }
    
    def draw_box(x, y, w, h, text, color, fontsize=8, bold=False):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", 
                            edgecolor='black', facecolor=color, linewidth=1.5)
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
               fontsize=fontsize, weight=weight)
    
    def draw_arrow(x1, y1, x2, y2, color='black', width=2):
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->',
                               color=color, linewidth=width, mutation_scale=15)
        ax.add_patch(arrow)
    
    # title
    ax.text(5, 11.3, 'Maskable PPO with BOARD_ONLY Feature Extractor', 
           ha='center', fontsize=16, weight='bold')
    ax.text(5, 10.9, '(CNN-only: processes board observations)', 
           ha='center', fontsize=10, style='italic', color='gray')
    
    # input
    draw_box(4, 10.2, 2, 0.5, 'Board Observation\n(20 channels)', colors['input'], 10, True)
    
    # cnn layers
    y_pos = 9.2
    x_center = 4
    
    # convolutional block 1
    draw_box(x_center, y_pos, 1.1, 0.35, 'Conv2d\n20→64, 3x3', colors['cnn'], 7)
    draw_box(x_center+1.2, y_pos, 0.8, 0.35, 'BN+ReLU', colors['activation'], 7)
    y_pos -= 0.45
    draw_box(x_center, y_pos, 1.1, 0.35, 'Conv2d\n64→64, 3x3', colors['cnn'], 7)
    draw_box(x_center+1.2, y_pos, 0.8, 0.35, 'BN+ReLU', colors['activation'], 7)
    y_pos -= 0.45
    draw_box(x_center+0.4, y_pos, 1.2, 0.3, 'MaxPool 2×2', colors['pooling'], 7)
    
    # conv block 2
    y_pos -= 0.45
    draw_box(x_center, y_pos, 1.1, 0.35, 'Conv2d\n64→128, 3x3', colors['cnn'], 7)
    draw_box(x_center+1.2, y_pos, 0.8, 0.35, 'BN+ReLU', colors['activation'], 7)
    y_pos -= 0.45
    draw_box(x_center, y_pos, 1.1, 0.35, 'Conv2d\n128→128, 3x3', colors['cnn'], 7)
    draw_box(x_center+1.2, y_pos, 0.8, 0.35, 'BN+ReLU', colors['activation'], 7)
    y_pos -= 0.45
    draw_box(x_center+0.4, y_pos, 1.2, 0.3, 'MaxPool 2×2', colors['pooling'], 7)
    
    # 3
    y_pos -= 0.45
    draw_box(x_center, y_pos, 1.1, 0.35, 'Conv2d\n128→256, 3x3', colors['cnn'], 7)
    draw_box(x_center+1.2, y_pos, 0.8, 0.35, 'BN+ReLU', colors['activation'], 7)
    y_pos -= 0.45
    draw_box(x_center, y_pos, 1.1, 0.35, 'Conv2d\n256→256, 3x3', colors['cnn'], 7)
    draw_box(x_center+1.2, y_pos, 0.8, 0.35, 'BN+ReLU', colors['activation'], 7)
    y_pos -= 0.45
    draw_box(x_center, y_pos, 1.1, 0.35, 'Conv2d\n256→256, 3x3', colors['cnn'], 7)
    draw_box(x_center+1.2, y_pos, 0.8, 0.35, 'BN+ReLU', colors['activation'], 7)
    y_pos -= 0.45
    draw_box(x_center+0.4, y_pos, 1.2, 0.3, 'AdaptiveAvg 3×2', colors['pooling'], 7)
    y_pos -= 0.4
    draw_box(x_center+0.4, y_pos, 1.2, 0.3, 'Flatten → 1536', colors['cnn'], 7)
    
    # fully connected layers
    y_pos -= 0.5
    draw_box(x_center-0.3, y_pos, 2.6, 0.4, 'Linear 1536→512\nLayerNorm, ReLU, Dropout', colors['cnn'], 8)
    y_pos -= 0.5
    draw_box(x_center-0.3, y_pos, 2.6, 0.4, 'Linear 512→512\nLayerNorm, ReLU', colors['cnn'], 8)
    
    # pi and vf networks
    draw_box(2.5, 1.5, 1.8, 0.35, 'Policy MLP\n512→256→128, Tanh', colors['policy'], 8)
    draw_box(2.5, 0.9, 1.8, 0.35, 'Action Net\n128→290', colors['output'], 9, True)
    
    draw_box(5.7, 1.5, 1.8, 0.35, 'Value MLP\n512→256→128, Tanh', colors['value'], 8)
    draw_box(5.7, 0.9, 1.8, 0.35, 'Value Net\n128→1', colors['output'], 9, True)
    
    # arrows
    draw_arrow(5, 3.6, 3.4, 1.85, '#FF6B6B', width=2)
    draw_arrow(5, 3.6, 6.6, 1.85, '#4ECDC4', width=2)
    
    # labels
    ax.text(3.4, 0.6, 'Action Logits (290)', ha='center', fontsize=8, style='italic')
    ax.text(6.6, 0.6, 'State Value', ha='center', fontsize=8, style='italic')
    
    plt.tight_layout()
    return fig


def visualise_numeric_only():
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    colors = {
        'input': '#E8F4F8',
        'mlp': '#B0E0B0',
        'policy': '#FF6B6B',
        'value': '#4ECDC4',
        'output': '#FFD700',
    }
    
    def draw_box(x, y, w, h, text, color, fontsize=9, bold=False):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", 
                            edgecolor='black', facecolor=color, linewidth=1.5)
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
               fontsize=fontsize, weight=weight)
    
    def draw_arrow(x1, y1, x2, y2, color='black', width=2):
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->',
                               color=color, linewidth=width, mutation_scale=15)
        ax.add_patch(arrow)
    
    # title
    ax.text(5, 9.3, 'Maskable PPO with NUMERIC_ONLY Feature Extractor', 
           ha='center', fontsize=16, weight='bold')
    ax.text(5, 8.9, '(MLP-only: processes numeric features)', 
           ha='center', fontsize=10, style='italic', color='gray')
    
    # input
    draw_box(3.5, 8.2, 3, 0.5, 'Numeric Observation (76 features)', colors['input'], 11, True)
    
    # MLP layers
    y_pos = 7.2
    draw_box(3, y_pos, 4, 0.5, 'Linear 76→256\nLayerNorm, ReLU, Dropout 0.1', colors['mlp'], 9)
    y_pos -= 0.7
    draw_box(3, y_pos, 4, 0.5, 'Linear 256→256\nLayerNorm, ReLU, Dropout 0.1', colors['mlp'], 9)
    y_pos -= 0.7
    draw_box(3, y_pos, 4, 0.5, 'Linear 256→512\nLayerNorm, ReLU', colors['mlp'], 9)
    y_pos -= 0.7
    draw_box(3, y_pos, 4, 0.5, 'Linear 512→512\nLayerNorm, ReLU', colors['mlp'], 9)
    
    # policy and value networks
    draw_box(1.8, 3.2, 2.4, 0.4, 'Policy MLP\n512→256→128, Tanh', colors['policy'], 9)
    draw_box(1.8, 2.5, 2.4, 0.4, 'Action Net\n128→290', colors['output'], 10, True)
    
    draw_box(5.8, 3.2, 2.4, 0.4, 'Value MLP\n512→256→128, Tanh', colors['value'], 9)
    draw_box(5.8, 2.5, 2.4, 0.4, 'Value Net\n128→1', colors['output'], 10, True)
    
    # arrows
    draw_arrow(5, 8.2, 5, 7.7, 'black', 2)
    draw_arrow(5, 5, 3, 3.6, '#FF6B6B', 2.5)
    draw_arrow(5, 5, 7, 3.6, '#4ECDC4', 2.5)
    draw_arrow(3, 3.2, 3, 2.9, '#FF6B6B')
    draw_arrow(7, 3.2, 7, 2.9, '#4ECDC4')
    
    # labels
    ax.text(3, 2.1, 'Action Logits (290)', ha='center', fontsize=9, style='italic')
    ax.text(7, 2.1, 'State Value', ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    return fig


def main():
    
    print("="*70)
    print("GENERATING ALL 4 MASKABLE PPO ARCHITECTURES")
    print("="*70)
    
    # 1. MLP Baseline (FlattenExtractor)
    print("\n1. Generating MLP baseline (FlattenExtractor)...")
    fig1 = visualise_mlp_flatten()
    fig1.savefig('bots/network_figs/mlp_baseline.png', dpi=300, bbox_inches='tight')
    print("Saved: mlp_baseline.png")
    
    # 2. COMBINED extractor
    print("\n2. Generating COMBINED feature extractor...")
    fig2 = visualise_combined_extractor()
    fig2.savefig('bots/network_figs/combined.png', dpi=300, bbox_inches='tight')
    print("Saved: combined.png")
    
    # 3. BOARD_ONLY extractor
    print("\n3. Generating BOARD_ONLY feature extractor...")
    fig3 = visualise_board_only()
    fig3.savefig('bots/network_figs/board_only.png', dpi=300, bbox_inches='tight')
    print("Saved: board_only.png")
    
    # 4. NUMERIC_ONLY extractor
    print("\n4. Generating NUMERIC_ONLY feature extractor...")
    fig4 = visualise_numeric_only()
    fig4.savefig('bots/network_figs/numeric_only.png', dpi=300, bbox_inches='tight')
    print("Saved: numeric_only.png")
    
    print("\n" + "="*70)
    print("ARCHITECTURE COMPARISON SUMMARY")
    print("="*70)
    
    print("\n1. MLP BASELINE (FlattenExtractor):")
    print("   Input: Flattened observation (1002 features)")
    print("   Architecture: Simple MLP")
    print("   Policy: 1002 → 256 → 128 (Tanh) → 290 actions")
    print("   Value: 1002 → 256 → 128 (Tanh) → 1 value")
    print("   Notes: Baseline model, no custom feature extraction")
    
    print("\n2. COMBINED Extractor:")
    print("   Input: Board (20 channels) + Numeric (76 features)")
    print("   CNN Path: 20→64→64→128→128→256→256→256 → Flatten(1536) → 256")
    print("   MLP Path: 76 → 256 → 256 → 256")
    print("   Combined: Concat[256+256=512] → 512 → 512")
    print("   Policy: 512 → 256 → 128 (Tanh) → 290 actions")
    print("   Value: 512 → 256 → 128 (Tanh) → 1 value")
    print("   Notes: Dual-path architecture for multi-modal input")
    
    print("\n3. BOARD_ONLY Extractor:")
    print("   Input: Board (20 channels)")
    print("   CNN: 20→64→64→128→128→256→256→256 → Flatten(1536) → 512 → 512")
    print("   Policy: 512 → 256 → 128 (Tanh) → 290 actions")
    print("   Value: 512 → 256 → 128 (Tanh) → 1 value")
    print("   Notes: CNN-only for spatial board features")
    
    print("\n4. NUMERIC_ONLY Extractor:")
    print("   Input: Numeric (76 features)")
    print("   MLP: 76 → 256 → 256 → 512 → 512")
    print("   Policy: 512 → 256 → 128 (Tanh) → 290 actions")
    print("   Value: 512 → 256 → 128 (Tanh) → 1 value")
    print("   Notes: MLP-only for tabular numeric features")
    
    print("\n" + "="*70)
    print("All visualizations generated successfully!")
    print("="*70)
    
    plt.show()


if __name__ == "__main__":
    main()