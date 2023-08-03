from utils import create_sets, check_dataset_count, check_image, get_class


def main():
    """
    RUN ONCE TO DIVIDE RAW IMAGES INTO TRAIN, VALID, AND TEST DATASETS
    """
    # Check that images have correct extension (i.e.,jpg, png, etc.)
    check_image("raw_images")

    create_sets(class_name='vangogh/', to_path='data',
                from_path='raw_images/vangogh/', pct_train=0.80, pct_test=0.15)
    create_sets(class_name='not_vangogh/', to_path='data',
                from_path='raw_images/not_vangogh/', pct_train=0.80, pct_test=0.15)

    # get file count for all the data folders
    check_dataset_count()

    # Checks class list
    get_class(file_path="data/valid/")


if __name__ == "__main__":
    main()