# tiny-llm

日本語の文章を文字単位で学習する、小さな Decoder-only Transformer です。LLM の仕組みを学ぶためのプロジェクトで、自然な会話や正確な文章の生成を目的としたものではありません。

## 必要なもの

- Docker Desktop（Compose を含む）
- CPU で実行します。Mac の Docker コンテナから Apple Silicon の MPS GPU は使用しません。

## 動かす

リポジトリのルートで以下を実行します。初回はイメージと依存関係のダウンロードに時間がかかります。

```bash
docker compose build
docker compose run --rm app pytest -q
docker compose run --rm app tiny-llm train --config configs/tiny.yaml
docker compose run --rm app tiny-llm generate --checkpoint artifacts/model.pt --prompt 'こんにちは' --length 100
```

学習中は `train_loss` と `val_loss` を表示します。値が小さいほど次の文字の予測誤差が小さいことを表します。サンプルは非常に小さく同じ文章も含むため、評価値や生成品質を一般的な性能の指標として扱えません。学習結果はホスト側の `artifacts/model.pt` に残ります。再学習すると同じパスのファイルは上書きされます。

生成時は `--temperature 0.8` のようにランダム性、`--seed 42` で乱数シードを指定できます。プロンプトには学習データに含まれる文字だけを使用できます。出力には改行が含まれることもあります。

## 自分のテキストを使う

UTF-8 のテキストを `data/` に配置し、`configs/tiny.yaml` をコピーして `data_path` と `output_path` を変更します。設定ファイルは `configs/` に置き、上記の `--config` にそのパスを指定してください。どちらもコンテナ内の `/app` からの相対パスです。`data/` と `configs/` は読み取り専用、`artifacts/` は書き込み可能としてホストから共有されます。利用権限のあるデータを使ってください。

テキストは先頭 90% を学習、残りを評価に使います。両方とも `block_size` より長くする必要があります。文字の対応表はテキスト全体から作り、モデルと一緒に保存します。未知文字用トークンはありません。モデルは推論用で、学習途中からの再開には対応していません。

## 開発と確認

```bash
docker compose run --rm app ruff check .
docker compose run --rm app pytest -q
```

依存関係は `uv.lock` で固定し、CI でテスト、静的チェック、開発用を含む Python 依存関係の脆弱性監査を実行します。Dependabot は uv、Docker、GitHub Actions の更新を確認します。既知の脆弱性の有無は時間とともに変わり、監査ですべてを検出できるわけではありません。監査では PyTorch の CPU ビルドの `+cpu` を取り除き、対応する PyPI のバージョンとして照合します。Docker ベースイメージの OS パッケージはこの監査の対象外です。

モデルの処理は [構造の説明](docs/architecture.md) を参照してください。学習やサンプリングの条件は `configs/tiny.yaml` と `tiny-llm generate --help` で確認できます。
