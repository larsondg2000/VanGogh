from predict_functions import parse_args, load_checkpoint, get_class, predict_image


def main():
    """
    checks individual image prediction

    :return:
    """
    args = parse_args()
    model = load_checkpoint(args.checkpoint)
    class_names = get_class(file_path=args.class_names)
    print(class_names)
    custom_image_path = "sample_paintings/surat.jpg"

    predict_image(model, custom_image_path, class_names, args.gpu)


if __name__ == "__main__":
    main()
