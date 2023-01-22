import dataclasses
from typing import List, Tuple, Any

import cv2
import numpy as np
import onnxruntime as ort


@dataclasses.dataclass
class HumanKeypoints:
    nose: Tuple[int, int]
    left_eye: Tuple[int, int]
    right_eye: Tuple[int, int]
    left_ear: Tuple[int, int]
    right_ear: Tuple[int, int]
    left_shoulder: Tuple[int, int]
    right_shoulder: Tuple[int, int]
    left_elbow: Tuple[int, int]
    right_elbow: Tuple[int, int]
    left_wrist: Tuple[int, int]
    right_wrist: Tuple[int, int]
    left_waist: Tuple[int, int]
    right_waist: Tuple[int, int]
    left_knee: Tuple[int, int]
    right_knee: Tuple[int, int]
    left_ankle: Tuple[int, int]
    right_ankle: Tuple[int, int]

    CONNECTION = [
        ["nose", "left_eye"],
        ["nose", "right_eye"],
        ["left_eye", "left_ear"],
        ["right_eye", "right_ear"],
        ["left_shoulder", "right_shoulder"],
        ["left_shoulder", "left_elbow"],
        ["left_elbow", "left_wrist"],
        ["right_shoulder", "right_elbow"],
        ["right_elbow", "right_wrist"],
        ["left_shoulder", "left_waist"],
        ["right_shoulder", "right_waist"],
        ["left_waist", "right_waist"],
        ["left_waist", "left_knee"],
        ["left_knee", "left_ankle"],
        ["right_waist", "right_knee"],
        ["right_knee", "right_ankle"],
    ]

    @classmethod
    def create_human_keypoint(cls, keypoints: List) -> "HumanKeypoints":
        """HumanKeypoinstsを作成する

        Args:
            keypoints (List): キーポイントリスト

        Returns:
            HumanKeypoints: HumanKeypointsオブジェクト
        """
        return cls(
            nose=(int(keypoints[0][0]), int(keypoints[0][1])),
            left_eye=(int(keypoints[1][0]), int(keypoints[1][1])),
            right_eye=(int(keypoints[2][0]), int(keypoints[2][1])),
            left_ear=(int(keypoints[3][0]), int(keypoints[3][1])),
            right_ear=(int(keypoints[4][0]), int(keypoints[4][1])),
            left_shoulder=(int(keypoints[5][0]), int(keypoints[5][1])),
            right_shoulder=(int(keypoints[6][0]), int(keypoints[6][1])),
            left_elbow=(int(keypoints[7][0]), int(keypoints[7][1])),
            right_elbow=(int(keypoints[8][0]), int(keypoints[8][1])),
            left_wrist=(int(keypoints[9][0]), int(keypoints[9][1])),
            right_wrist=(int(keypoints[10][0]), int(keypoints[10][1])),
            left_waist=(int(keypoints[11][0]), int(keypoints[11][1])),
            right_waist=(int(keypoints[12][0]), int(keypoints[12][1])),
            left_knee=(int(keypoints[13][0]), int(keypoints[13][1])),
            right_knee=(int(keypoints[14][0]), int(keypoints[14][1])),
            left_ankle=(int(keypoints[15][0]), int(keypoints[15][1])),
            right_ankle=(int(keypoints[16][0]), int(keypoints[16][1])),
        )

    def draw(self, bgr_image: np.ndarray, color_code: Tuple = (0, 255, 0)) -> np.ndarray:
        """キーポイントを描画する

        Args:
            bgr_image (np.ndarray): 入力画像
            color_code (Tuple, optional): 描画色コード. Defaults to (0, 255, 0).

        Returns:
            np.ndarray: 描画済み画像
        """
        varianse = vars(self)

        for _, pos in varianse.items():
            if pos[0] > 0 and pos[1] > 0:
                cv2.circle(
                    bgr_image,
                    (pos[0], pos[1]),
                    3,
                    color_code,
                    -1,
                    lineType=cv2.LINE_AA,
                )

        for connect in self.CONNECTION:
            x1, y1 = varianse[connect[0]][0], varianse[connect[0]][1]
            x2, y2 = varianse[connect[1]][0], varianse[connect[1]][1]

            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                cv2.line(
                        bgr_image,
                        (x1, y1),
                        (x2, y2),
                        color_code,
                        2,
                        lineType=cv2.LINE_AA,
                    )
        return bgr_image


class E2Pose:
    def __init__(self, model_path: str, thr: float) -> None:
        """Initialize

        Args:
            model_path (str): モデルファイルパス
            thr (float): 閾値
        """
        self._thr = thr
        self._sess = ort.InferenceSession(
            model_path,
            providers=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        )
        self._input_name = self._sess.get_inputs()[0].name
        self._input_shape = self._sess.get_inputs()[0].shape


    def process(self, bgr_image: np.ndarray) -> List[HumanKeypoints]:
        """推論する

        Args:
            bgr_image (np.ndarray): 入力画像

        Returns:
            List[HumanKeypoints]: HumanKeypoinstsのリスト
        """
        h, w, _ = bgr_image.shape
        input_image = self._preprocess(bgr_image)
        results = self._sess.run(None, {self._input_name: input_image})
        human_list = self._postprocess(results, w, h)
        return human_list

    def _preprocess(self, bgr_image: np.ndarray) -> np.ndarray:
        """前処理

        Args:
            bgr_image (np.ndarray): 入力画像

        Returns:
            np.ndarray: 前処理後画像
        """
        input_image = cv2.resize(bgr_image, dsize=(self._input_shape[3], self._input_shape[2]))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, axis=0)
        input_image = input_image.astype('float32')
        return input_image

    def _postprocess(self, results: List, original_width: int, original_height: int) -> List[HumanKeypoints]:
        """後処理

        Args:
            results (List): 推論結果
            original_width (int): 入力画像の幅
            original_height (int): 入力画像の高さ

        Returns:
            List[HumanKeypoints]: HumanKeypoinstsのリスト
        """
        human_list: List[HumanKeypoints] = []

        kpt, pv = results
        pv = np.reshape(pv[0], [-1])
        kpt = kpt[0][pv >= self._thr]
        kpt[:, :, -1] *= original_height
        kpt[:, :, -2] *= original_width
        kpt[:, :, -3] *= 2

        for human in kpt:
            mask = np.stack(
                [(human[:, 0] >= self._thr).astype(np.float32)],
                axis=-1,
            )
            human *= mask
            keypoints = np.stack([human[:, _ii] for _ii in [1, 2, 0]], axis=-1)

            human_list.append(
                HumanKeypoints.create_human_keypoint(keypoints)
            )
        return human_list
