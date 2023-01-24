# e2pose_example

## Overview
E2PoseのPythonでのONNX推論サンプルです。

## Environment
- Windows 10 Home
- Poetry

## Usage
### 環境構築
```
$poetry install --no-root
```

Poetry環境がない場合は以下を実行する。

```
$pip install -r requirements.txt
```

### 実行方法

PINTOさんの[PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo)からE2Poseのモデルをダウンロードする。

ダウンロードが完了したら以下コマンドを実行する。

```shell
$poetry run python -m e2pose --model {モデルパス}

# 以下コマンドオプション一覧
$poetry run python -m e2pose -h
usage: __main__.py [-h] [--device DEVICE] [--video VIDEO] [--model MODEL] [--thr THR]

optional arguments:
  -h, --help       show this help message and exit
  --device DEVICE  カメラデバイスID
  --video VIDEO    動画ファイルのパス（指定された場合これが優先される）
  --model MODEL    モデルファイルのパス
  --thr THR        検出閾値
```

## Author
T-Sumida

## Reference
- [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo)

## License
[Apache-2.0 license](https://github.com/T-Sumida/e2pose_example/blob/main/LICENSE)