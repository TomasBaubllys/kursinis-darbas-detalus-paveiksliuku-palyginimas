import argparse
import time

import torch

from dataset import Market_Train_Dataset
from models import MobileNetV3_BoT, ResNet18_BoT


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process model settings.")

    parser.add_argument(
        "-mbnet",
        "--mobilenetv3",
        action="store_const",
        const="mobilenetv3",
        default="resnet18",
        dest="model",
        help="use mobilenetv3 instead of resnet 18",
    )

    parser.add_argument("-wf", "--weights_file", type=str, help="Weights file to use")

    parser.add_argument(
        "-b",
        "--bot",
        type=int,
        help="BoT level that must match the level, the model was trained on",
    )

    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="default",
        help='device "cpu", "gpu" (if cuda available)',
    )

    return parser.parse_args()


def fps_test(
    model_name="resnet18", bot=3, weights_file="checkpoint.pth", _device="default"
):
    if _device == "default":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif _device == "cpu":
        device = torch.device("cpu")
    elif _device == "gpu":
        device = torch.device("cuda")

    print(f"using device {device}")

    img = torch.randn(1, 3, 256, 128).to(device)

    # grab num classes (it is needed for some bot levels)
    train_ds = Market_Train_Dataset("../data//Market-1501-v15.09.15/bounding_box_train")
    num_classes = train_ds.num_ids

    if model_name == "resnet18":
        model = ResNet18_BoT(num_classes=num_classes, bot_level=bot)
    elif model_name == "mobilenetv3":
        model = MobileNetV3_BoT(num_classes=num_classes, bot_level=bot)

    weights = torch.load(weights_file, map_location=device)
    model.load_state_dict(weights)

    model.to(device)

    model.eval()
    reps = 300
    start_time = time.time()
    with torch.no_grad():
        for _ in range(reps):
            _ = model(img)
    end_time = time.time()

    total_time = end_time - start_time
    ms_per_img = 1000.0 * total_time / reps

    print(f"Total time = {total_time}, ms/img = {ms_per_img} fps = {1000 / ms_per_img}")


if __name__ == "__main__":
    args = parse_arguments()
    fps_test(
        model_name=args.model,
        bot=args.bot,
        weights_file=args.weights_file,
        _device=args.device,
    )
