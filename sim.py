import matplotlib.pyplot as plt
from src.mirror import Mirror


def main():
    print("Hello Makyoh sim!")

    # Init:
    resolution = [4096, 4096]
    mirror_size = [1.0, 1.0]
    distance = 5.0
    mirror = Mirror(resolution, mirror_size, distance)

    # Render:
    heightmap = mirror.render_heightmap().T
    plt.figure()
    plt.title("Mirror heightmap")
    p = plt.imshow(heightmap)
    plt.colorbar(p)
    plt.show()

    canvas = mirror.render_canvas().T
    plt.figure()
    plt.title("Canvas image")
    p = plt.imshow(canvas)
    plt.colorbar(p)
    plt.show()


if __name__ == "__main__":
    main()

