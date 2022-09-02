from .dsfunction import imread, denormalize, reduce_to_scale
from . import dstransform as transforms
from .video import get_read_stream, get_writer_stream
from pathlib import Path
import cv2
from .terminfo import INFO
from more_itertools import chunked
import argparse
from tqdm import tqdm
import torch


def infer_fn(model, args_list):

    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--image_path", help="input path, can be image,vedio,directory", type=str
    )
    parse.add_argument(
        "--device",
        help="infer device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
    )
    parse.add_argument("--batch_size", help="infer batch size", type=int, default=16)
    args = parse.parse_args(args_list)

    device = torch.device(args.device)

    resize2scale = transforms.ResizeToScale((256, 256), 32)
    infer_transform = transforms.Compose(
        [
            resize2scale,
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    model.setup("test")
    model.eval()

    model = model.to(device)

    def infer_batch(feed_im: torch.Tensor):
        feed_im = feed_im.to(device)
        out_im = model.forward(feed_im)
        draw_im = (
            denormalize(out_im.permute((0, 2, 3, 1)).detach().to("cpu").numpy()) * 255
        ).astype("uint8")
        return draw_im

    def infer_one_image(image_path: Path, output_root: Path):
        im = imread(image_path.as_posix())
        feed_im = infer_transform(im)
        draw_im = infer_batch(feed_im[None, ...])
        draw_im = draw_im[0]
        output_path = output_root / (image_path.stem + "_out" + image_path.suffix)
        cv2.imwrite(output_path.as_posix(), cv2.cvtColor(draw_im, cv2.COLOR_RGB2BGR))
        print(INFO, "Convert", image_path, "to", output_path)

    def infer_video(video_path: Path, output_root: Path, batch_size=16):
        read_stream, length, fps, height, width = get_read_stream(video_path)
        height, width = reduce_to_scale(
            [height, width], resize2scale.size[::-1], resize2scale.scale
        )
        output_path = output_root / (image_path.stem + "_out" + ".mp4")
        writer_stream = get_writer_stream(output_path, fps, height, width)
        for frames in tqdm(
            chunked(read_stream, batch_size), total=length // batch_size
        ):
            feed_im = torch.stack(
                [
                    infer_transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    for frame in frames
                ]
            )
            draw_im = infer_batch(feed_im)
            for im in draw_im:
                writer_stream.write(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        writer_stream.release()

    image_path = Path(args.image_path)
    if image_path.is_file():
        output_root = image_path.parent
        if image_path.suffix in [".mp4", ".flv"]:
            infer_video(image_path, output_root, args.batch_size)
        else:
            infer_one_image(image_path, output_root)
    elif image_path.is_dir():
        output_root: Path = image_path.parent / (image_path.name + "_out")
        if not output_root.exists():
            output_root.mkdir()
        for p in image_path.iterdir():
            infer_one_image(p, output_root)
