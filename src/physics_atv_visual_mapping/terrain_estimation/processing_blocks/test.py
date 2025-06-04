import argparse
import matplotlib.pyplot as plt

from physics_atv_visual_mapping.terrain_estimation.processing_blocks.utils import sobel_x_kernel, sobel_y_kernel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, required=True)
    args = parser.parse_args()

    fig, axs = plt.subplots(1, 2)

    axs[0].imshow(sobel_x_kernel(rad=[args.n, args.n]))
    axs[1].imshow(sobel_y_kernel(rad=[args.n, args.n]))

    axs[0].set_title('sobel x')
    axs[1].set_title('sobel y')

    plt.show()
