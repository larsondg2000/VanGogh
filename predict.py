from predict_functions import parse_args, load_checkpoint, get_class, check_valid


def main():
    """
    Checks validation image predictions
    :return:
    """
    args = parse_args()
    model = load_checkpoint(args.checkpoint)
    class_names = get_class(file_path=args.class_names)

    check_valid(filepath="data/valid/",
                model=model,
                class_names=class_names,
                gpu=args.gpu)


if __name__ == "__main__":
    main()
