import matplotlib.pyplot as plt
import src.core_sim as core_sim
import src.mirror as mirror


def main():
    print("Hello Makyoh sim!")
    resolution = 1000
    my_mirror = mirror.Mirror()
    canvas_image = core_sim.render_reflection(resolution, my_mirror)

    # show hight map in 2d
    plt.figure()
    plt.title("Reflected image on the canvas")
    p = plt.imshow(canvas_image)
    plt.colorbar(p)
    plt.show()


if __name__ == "__main__":
    main()
