from predict_functions import predict_random, get_class, load_checkpoint, parse_args


def main():
    """
    Randomly selects a number of images from a filepath and returns predictions and
    shows the image

    :return:
    """
    args = parse_args()
    model = load_checkpoint(args.checkpoint)
    class_names = get_class(file_path=args.class_names)

    predict_random(filepath="data/valid/",
                   model=model,
                   class_names=class_names,
                   gpu=args.gpu)


if __name__ == "__main__":
    main()
