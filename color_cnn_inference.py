import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from color_distillation.models.color_cnn import ColorCNN
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
import numpy as np

def main():
    # settings
    parser = argparse.ArgumentParser(description='ColorCNN inference')
    parser.add_argument('--num_colors', type=int, default=16)
    parser.add_argument('--alpha', type=float, default=1, help='multiplier of regularization terms')
    parser.add_argument('--beta', type=float, default=0, help='multiplier of regularization terms')
    parser.add_argument('--gamma', type=float, default=0, help='multiplier of reconstruction loss')
    parser.add_argument('--color_jitter', type=float, default=1)
    parser.add_argument('--color_norm', type=float, default=4, help='normalizer for color palette')
    parser.add_argument('--label_smooth', type=float, default=0.0)
    parser.add_argument('--soften', type=float, default=1, help='soften coefficient for softmax')
    parser.add_argument('--backbone', type=str, default='unet', choices=['unet', 'dncnn'])
    args = parser.parse_args()

    model = ColorCNN(args.backbone, args.num_colors, args.soften, args.color_norm, args.color_jitter).cuda()
    model.load_state_dict(torch.load('ColorCNN.pth'))
    model.eval()

    original_image = Image.open('parrot.jpg').convert('RGB')
    img = ToTensor()(original_image).unsqueeze(0).cuda()
    transformed_img = model(img, training=False)
    reduced_image = torch.Tensor.permute(torch.squeeze(transformed_img[0]), (1, 2, 0)).cpu()
    palette = np.array(torch.Tensor.permute(torch.squeeze(transformed_img[2]), (1, 0)).cpu())
    cmp = ListedColormap(palette)

    fig, axis = plt.subplots(2, 2)
    axis[0, 0].imshow(original_image)
    axis[0, 0].axis('off')
    axis[0, 0].title.set_text('Original Image')
    axis[0, 1].imshow(reduced_image)
    axis[0, 1].axis('off')
    axis[0, 1].title.set_text('16-color Image using ColorCNN')

    Z = np.array(np.arange(1, 17)).reshape(1, 16)
    x = np.arange(0.5, 17, 1)  # len = 11
    y = np.arange(0, 1.5, 1)  # len = 1
    axis[1, 0].axis('off')
    axis[1, 1].pcolormesh(x, y, Z, cmap=cmp)
    axis[1, 1].get_yaxis().set_visible(False)
    axis[1, 1].title.set_text('16 color palette')
    axis[1, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
