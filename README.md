# llamafit

**llama.cpp の推論パラメータ決めを補助する CLI ツール**です。GGUF のヘッダ（メタデータ）だけを読んで、

- モデル重みと KV キャッシュが **VRAM にどこまで乗るか**を可視化
- llama.cpp の **`--fit`（自動フィット）が選びそうなパラメータ**をシミュレーションして表示
- ターゲットのコンテキスト長を守るための **推奨パラメータ**（`-ngl` / `-c` / `-ctk`・`-ctv` の KV 量子化 / MoE 向け `--n-cpu-moe`）と、そのまま貼れるコマンドラインを提示

します。テンソルデータは読まないので、100GB のモデルでも一瞬で解析できます（リモート URL なら数 MB のダウンロードだけ）。

```
$ uvx --from git+https://github.com/ilolio/visualize-llamacpp-speed llamafit ~/models/Qwen3-32B-Q4_K_M.gguf
```

## なぜ CLI？

llama.cpp を動かす GPU マシンはたいていヘッドレスで SSH 越しに触るものなので、ブラウザではなく**ターミナルでそのまま完結する**ことを優先しました。`uv tool run`（`uvx`）は CLI ツールをインストールなしで実行する仕組みそのものです。スクリプトから使う場合は `--json` で機械可読な出力も得られます。

## インストール / 実行

```bash
# その場で実行（インストール不要）
uvx --from git+https://github.com/ilolio/visualize-llamacpp-speed llamafit MODEL.gguf

# ツールとしてインストール
uv tool install git+https://github.com/ilolio/visualize-llamacpp-speed
llamafit MODEL.gguf
```

依存は `rich` のみ。Python 3.10+。

## 使い方

```bash
# GPU マシン上なら VRAM を自動検出（nvidia-smi / rocm-smi / Apple Metal）
llamafit ~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

# ターゲットのコンテキスト長を指定
llamafit model.gguf -c 65536

# GPU の無いマシンから、24GB の GPU を想定して試算
llamafit model.gguf --vram 24

# 分割 GGUF はどれか 1 つのシャードを渡せば残りも自動で読む
llamafit model-00001-of-00005.gguf

# Hugging Face 上のモデルをダウンロードせずに解析（Range リクエストでヘッダのみ取得）
# llama.cpp の -hf と同じ書き方が使える（量子化タグでファイルを自動選択）
llamafit unsloth/Qwen3.5-9B-GGUF:Q4_K_M
llamafit unsloth/Qwen3.5-9B-GGUF          # タグ省略時は llama.cpp と同じく Q4_K_M → Q8_0 の順で探す
llamafit hf:bartowski/Qwen2.5-32B-Instruct-GGUF/Qwen2.5-32B-Instruct-Q4_K_M.gguf  # ファイル直接指定
# ゲート付きリポジトリは HF_TOKEN を設定

# パラメータを固定して「この設定なら乗るか？」を確認
llamafit model.gguf -c 32768 --ngl 40 --ctk q8_0 --vram 16

# スクリプト向け
llamafit model.gguf --vram 24 --json | jq .recommendations
```

### 主なオプション

| オプション | 意味 | デフォルト |
|---|---|---|
| `-c, --ctx` | ターゲットのコンテキスト長 | min(学習時 ctx, 32768) |
| `--vram GIB` | VRAM 予算(GiB)。未指定なら自動検出 | 自動検出 |
| `--ngl` | `-ngl` を固定（fit を無効化） | 自動 |
| `--ctk / --ctv` | KV キャッシュ型を固定 (`f16` `q8_0` `q4_0` など) | f16 |
| `--n-cpu-moe` | 先頭 N 層のエキスパートを CPU に固定 | 0 |
| `--fit-target MIB` | 残しておく空き VRAM（llama.cpp と同じ意味） | 1024 |
| `--fit-ctx N` | fit シミュレーションが縮めてよい ctx の下限 | 4096 |
| `--overhead MIB` | CUDA コンテキスト等のランタイムオーバーヘッド見積り | 500 |
| `--fa on/off` | Flash Attention の想定 | on |
| `--json` | JSON 出力 | — |

## 出力例

Qwen3-30B-A3B (Q4_K_M) を 16GB GPU / ctx 32768 で解析した例（抜粋・実際はカラー表示）:

```
Model memory needed  weights 18.39 GiB + KV 3.00 GiB @ ctx 32,768 = 21.39 GiB
████████████████████████████████████████████████████████████
                                         ▲ budget 15.00 GiB
■ weights 18.39 GiB   ■ KV f16 3.00 GiB

GPU usage after CPU offload  14.76 GiB used / 15.00 GiB budget   ✓ fits, 241 MiB headroom
███████████████████████████████████████████████████████████░
■ weights 10.95 GiB  ■ KV f16 3.00 GiB  ■ compute* 329 MiB  ■ overhead 500 MiB  ░ free 241 MiB

CPU RAM needed  7.47 GiB / 15.14 GiB available (15.70 GiB total)   ✓ fits, 7.67 GiB headroom
██████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
■ weights 7.44 GiB   ■ buffers 32 MiB   ░ free 7.67 GiB

layers  -ngl 49: 48/48 blocks on GPU +output
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█████████████████████████████
█ GPU   ▓ dense on GPU, experts on CPU   ░ CPU

╭────── llama.cpp --fit simulation (what auto-fit would pick) ──────╮
│ · expert tensors of first 20 layers moved to CPU (--n-cpu-moe 20) │
│ -c 32,768   -ngl 49   --n-cpu-moe 20   KV f16/f16                 │
╰───────────────────────────────────────────────────────────────────╯

Recommendations
╭──────────── ★ MoE experts on CPU (--n-cpu-moe 16) + q8_0 KV ────────────╮
│ $ llama-server -m Qwen3-30B-A3B-Q4_K_M.gguf -c 32768 -ngl 49 -fa on \   │
│     -ctk q8_0 -ctv q8_0 --n-cpu-moe 16                                  │
╰─────────────────────────────────────────────────────────────────────────╯
```

## 出力の見方

1. **モデル情報** — アーキテクチャ、量子化、層数、GQA/SWA/MoE/MLA などメモリに効く特徴
2. **必要メモリバー** — まず素の必要量（重み + KV キャッシュの合計 GiB）。予算超過なら予算位置をマーカー表示。GPU 未検出でも表示されます
3. **GPU 使用バー** — CPU オフロード（fit の調整）適用後の実際の GPU メモリ内訳（重み / KV / 計算バッファ / オーバーヘッド / 空き）
4. **CPU RAM バー** — オフロードの結果ホスト RAM 側に乗る分（重み / KV / ホストバッファ）を、搭載 RAM の空き容量と比較。足りなければスワップ警告
5. **レイヤーストリップ** — どの層が GPU に乗るか（MoE はエキスパートだけ CPU の層も区別）
6. **KV キャッシュ表** — `f16` / `q8_0` / `q4_0` それぞれの KV サイズ、フルオフロードで届く最大 ctx、ターゲット ctx での最大 `-ngl`
7. **--fit シミュレーション** — llama.cpp の自動フィット（デフォルト有効）が行う調整の予測: ①そのまま乗るか → ②ctx を縮小（`-c` 明示時はスキップ）→ ③MoE エキスパートを CPU へ → ④`-ngl` を削減
8. **推奨設定** — ctx を守る前提でのランク付き推奨。★付きが第一候補で、コピペ可能な `llama-server` コマンド付き

## 計算の中身と精度

- **重み**: GGUF のテンソルテーブルから正確に算出。`token_embd` は常に CPU、`-ngl` は最後の N ブロック + 出力層（`ngl > n_layer` のとき）という llama.cpp の配置を再現
- **KV キャッシュ**: `ctx × Σ層 (K次元 × K型のbytes/elem + V次元 × V型のbytes/elem)`。GQA の per-layer KV ヘッド数、SWA 層（gemma2/3, gpt-oss, cohere2 等は窓+マイクロバッチ分のみ）、deepseek2 系 MLA（潜在 KV、V キャッシュなし）に対応
- **計算バッファ**: 見積り（±15% 程度）。ロジット（`n_vocab × n_ubatch × f32`）、FA 無効時の KQ 行列、FFN 活性が主成分。だからこそ llama.cpp と同じく `--fit-target` の安全マージンを確保します
- **KV 量子化の推奨**: `q8_0` は実用上ほぼ無劣化で KV 半減。`q4_0` はさらに半分になるが品質低下と長 ctx での速度低下があるため、第一候補には出しません。fused Flash Attention の速いカーネルを使うため **K/V は同じ型**（対称）を推奨します

## 開発

```bash
uv sync
uv run pytest
```

## ライセンス

MIT
