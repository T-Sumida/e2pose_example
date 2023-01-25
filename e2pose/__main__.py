import argparse
import copy
import time
import logging

import cv2

from e2pose.e2pose import E2Pose


def get_args() -> argparse.Namespace:
    """引数情報を取得

    Returns:
        argparse.Namespace: 引数情報
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0, help="カメラデバイスID")
    parser.add_argument("--video", type=str, default=None, help="動画ファイルのパス（指定された場合これが優先される）")
    parser.add_argument("--model", type=str, default="e2epose_resnet50_1x3x512x512.onnx", help="モデルファイルのパス")
    parser.add_argument("--thr", type=float, default=0.5, help="検出閾値")

    args = parser.parse_args()

    return args


def main(args: argparse.Namespace) -> None:
    """推論

    Args:
        args (argparse.Namespace): 引数情報
    """
    e2pose = E2Pose(args.model, args.thr)

    if args.video is not None:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(args.device)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            start_t = time.time()

            debug_image = copy.deepcopy(frame)
            human_list = e2pose.process(debug_image)

            elapsed_time = time.time() - start_t

            # 結果を描画
            for human in human_list:
                human.draw(debug_image, (82, 232, 175))
            
            text = f"Elapsed Time: {(elapsed_time * 1000):.1f}[ms]"
            debug_image = cv2.putText(
                debug_image,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                thickness=2,
            )
            
            cv2.imshow("e2pose example", debug_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        logging.info("press ctrl+c")
    except Exception as e:
        logging.error(e)
    finally:
        cap.release()


if __name__ == "__main__":
    args = get_args()
    main(args)
