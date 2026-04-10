from generate_dataset import main as generate_dataset_main
from train_recursive_stressnet import main as train_recursive_stressnet_main


def main() -> None:
    generate_dataset_main()
    train_recursive_stressnet_main()


if __name__ == "__main__":
    main()
