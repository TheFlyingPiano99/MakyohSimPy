import matplotlib.pyplot as plt
from src.mirror import Mirror


def main():
    print("Hello Makyoh sim!")
    resolution = [1000, 1000]
    my_mirror = Mirror()
    canvas_image = my_mirror.render_canvas(resolution)

    # show hight map in 2d
    plt.figure()
    plt.title("Reflected image on the canvas")
    p = plt.imshow(canvas_image)
    plt.colorbar(p)
    plt.show()


if __name__ == "__main__":
    main()
