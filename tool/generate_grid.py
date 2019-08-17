from pathlib import Path
from random import sample
from argparse import ArgumentParser
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torchvision.utils import save_image

def sample_images(image_root, num_sampled):
    p = Path(image_root)
    images = list(p.glob("*.jpg"))
    sampled_imgs = sample(images, num_sampled)
    return sampled_imgs

def read_images(images_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    loader = default_loader

    return [transform(loader(img)) for img in images_path]

if __name__ == '__main__':
    parser = ArgumentParser("generate a grid image contained sampled generated images")
    parser.add_argument("-r", "--root", type=str, help="the folder that contains all generated image", required=True)
    parser.add_argument("-o", "--output", type=str, help="output image path", default="./PG2-origin.jpg")
    opt = parser.parse_args()
    save_image(read_images(sample_images(opt.root, 20)), opt.output, nrow=5, padding=0, normalize=True)