import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import numpy as np
from utils import setup_matplotlib, get_result_dir, load_model, get_test_loader, DATASETS
import config


class GradCAMHook:
    def __init__(self):
        self.activations = None
        self.gradients = None

    def forward_hook(self, module, input, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()


def register_hooks(model):
    hooks = {}
    hook_handles = []

    print_stage = model.backbone.print_backbone.stage5
    vein_stage = model.backbone.vein_backbone.stage5

    for name, module in [('print', print_stage), ('vein', vein_stage)]:
        hook = GradCAMHook()
        h_fwd = module.register_forward_hook(hook.forward_hook)
        h_bwd = module.register_full_backward_hook(hook.backward_hook)
        hooks[name] = hook
        hook_handles.extend([h_fwd, h_bwd])

    return hooks, hook_handles


def remove_hooks(hook_handles):
    for h in hook_handles:
        h.remove()


def compute_cam(activations, gradients):
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = cam.squeeze(0).squeeze(0).cpu().numpy()
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
    return tensor * std + mean


def overlay_heatmap(img_np, cam, alpha=0.5):
    plt = setup_matplotlib()
    import cv2

    h, w = img_np.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.astype(np.float32) / 255.0

    overlay = (1 - alpha) * img_np + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    return overlay


def generate_gradcam(model, print_img, vein_img, device, hooks):
    model.zero_grad()

    print_img = print_img.to(device)
    vein_img = vein_img.to(device)

    logits = model(print_img, vein_img)
    pred = logits.argmax(dim=1)

    target_score = logits[0, pred[0]]
    target_score.backward(retain_graph=True)

    print_cam = compute_cam(hooks['print'].activations, hooks['print'].gradients)
    vein_cam = compute_cam(hooks['vein'].activations, hooks['vein'].gradients)

    return print_cam, vein_cam, pred.item()


def collect_samples(model, loader, device, hooks, num_correct, num_incorrect):
    correct_samples = []
    incorrect_samples = []

    mean = config.NORMALIZE_MEAN
    std = config.NORMALIZE_STD

    for print_img, vein_img, labels in loader:
        for i in range(print_img.size(0)):
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                return correct_samples, incorrect_samples

            p_img = print_img[i:i+1].to(device)
            v_img = vein_img[i:i+1].to(device)
            label = labels[i].item()

            p_cam, v_cam, pred = generate_gradcam(model, p_img, v_img, device, hooks)

            p_denorm = denormalize(print_img[i], mean, std)
            v_denorm = denormalize(vein_img[i], mean, std)

            p_np = p_denorm.permute(1, 2, 0).cpu().numpy()
            v_np = v_denorm.permute(1, 2, 0).cpu().numpy()
            p_np = np.clip(p_np, 0, 1)
            v_np = np.clip(v_np, 0, 1)

            sample = {
                'print_img': p_np,
                'vein_img': v_np,
                'print_cam': p_cam,
                'vein_cam': v_cam,
                'pred': pred,
                'label': label,
            }

            if pred == label and len(correct_samples) < num_correct:
                correct_samples.append(sample)
            elif pred != label and len(incorrect_samples) < num_incorrect:
                incorrect_samples.append(sample)

    return correct_samples, incorrect_samples


def plot_samples(samples, save_dir, prefix, alpha):
    plt = setup_matplotlib()

    for idx, sample in enumerate(samples):
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

        p_overlay = overlay_heatmap(sample['print_img'], sample['print_cam'], alpha)
        v_overlay = overlay_heatmap(sample['vein_img'], sample['vein_cam'], alpha)

        axes[0, 0].imshow(sample['print_img'])
        axes[0, 0].set_title('Palmprint')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(sample['print_cam'], cmap='jet')
        axes[0, 1].set_title('Palmprint CAM')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(p_overlay)
        axes[0, 2].set_title('Palmprint Overlay')
        axes[0, 2].axis('off')

        axes[1, 0].imshow(sample['vein_img'])
        axes[1, 0].set_title('Palm Vein')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(sample['vein_cam'], cmap='jet')
        axes[1, 1].set_title('Palm Vein CAM')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(v_overlay)
        axes[1, 2].set_title('Palm Vein Overlay')
        axes[1, 2].axis('off')

        match = 'Correct' if sample['pred'] == sample['label'] else 'Incorrect'
        fig.suptitle(
            f'{match} - Pred: {sample["pred"]}, GT: {sample["label"]}',
            fontsize=14
        )

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{prefix}_{idx:03d}.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Grad-CAM Visualization for VIBE-Net')
    parser.add_argument('--dataset', type=str, default='QH', choices=DATASETS)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--num-correct', type=int, default=5)
    parser.add_argument('--num-incorrect', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.5)
    args = parser.parse_args()

    model, checkpoint, device = load_model(args.dataset, args.checkpoint)
    loader = get_test_loader(args.dataset, batch_size=args.batch_size)

    hooks, hook_handles = register_hooks(model)

    save_dir = get_result_dir(args.dataset, 'gradcam')

    print(f'Collecting Grad-CAM samples for dataset: {args.dataset}')
    print(f'Saving to: {save_dir}')

    correct_samples, incorrect_samples = collect_samples(
        model, loader, device, hooks, args.num_correct, args.num_incorrect
    )

    print(f'Collected {len(correct_samples)} correct and {len(incorrect_samples)} incorrect samples')

    if correct_samples:
        plot_samples(correct_samples, save_dir, 'correct', args.alpha)
        print(f'Saved {len(correct_samples)} correct prediction visualizations')

    if incorrect_samples:
        plot_samples(incorrect_samples, save_dir, 'incorrect', args.alpha)
        print(f'Saved {len(incorrect_samples)} incorrect prediction visualizations')

    remove_hooks(hook_handles)
    print('Done.')


if __name__ == '__main__':
    main()
