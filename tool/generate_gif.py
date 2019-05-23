import imageio


def generate_gif(gif_path, filenames, duration=0.1):
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))

    imageio.mimsave(gif_path, images, duration=duration)


def main():
    path = "./checkpoints/PG2-improve-16/"
    filenames = [
        "{}iteration_{}.png".format(path, i * 200)
        for i in range(1, 38)
    ]

    generate_gif("PG2-improve-16.gif", filenames, 0.1)


if __name__ == '__main__':
    main()
