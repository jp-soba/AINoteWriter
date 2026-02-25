# AINoteWriter

X の Community Notes AI Writer を、以下 2 つの実行経路に対応させた実装です。

1. GitHub Actions から定期/手動実行
2. デスクトップアプリ（Tkinter GUI）から実行・結果確認

参考:
- X API Community Notes: https://docs.x.com/x-api/community-notes/introduction
- Template API Note Writer: https://github.com/twitter/communitynotes/tree/main/template-api-note-writer

---

## 機能

- `posts_eligible_for_notes` の取得
  - `post_selection=feed_lang:ja` で固定
- AI でノート下書き生成（OpenAI 互換エンドポイント）
- `evaluate_note` で事前評価（任意）
- `notes` への投稿（任意）
- `notes_written` 取得
- 実行結果を JSON で保存（`outputs/`）

---

## 前提

- Python 3.10+
- X API/Community Notes AI Writer の利用資格
- X API キー（ユーザーコンテキスト）
  - `X_API_KEY`
  - `X_API_KEY_SECRET`
  - `X_ACCESS_TOKEN`
  - `X_ACCESS_TOKEN_SECRET`
- AI API キー（任意。下書きを生成する場合）
  - `AI_API_KEY`

---

## セットアップ

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

`.env.example` を `.env` にコピーして値を設定してください。

---

## CLI 実行

### 1) 下書きのみ（投稿しない）

```bash
ai-note-writer run --num-posts 5 --test-mode true --submit-notes false --evaluate-before-submit true
```

### 2) 投稿あり（test_mode=true のまま推奨）

```bash
ai-note-writer run --num-posts 5 --test-mode true --submit-notes true --evaluate-before-submit true
```

### 3) 自分が書いたノートの確認

```bash
ai-note-writer notes --max-results 20
```

実行ログ JSON は `outputs/runs/` または `outputs/notes/` に保存されます。

---

## デスクトップアプリ（GUI）

```bash
ai-note-writer-gui
```

- `Run writer`: 取得→生成→(評価)→(投稿) ※ 日本語フィード固定
- `Fetch notes_written`: 直近の投稿ノート情報取得
- `Open last JSON`: 最新の結果ファイルを開く

---

## GitHub Actions

ワークフロー: `.github/workflows/community_note_writer.yml`

GitHub リポジトリの `Settings > Secrets and variables > Actions` で以下を設定:

- `X_API_KEY`
- `X_API_KEY_SECRET`
- `X_ACCESS_TOKEN`
- `X_ACCESS_TOKEN_SECRET`
- `AI_API_KEY`
- 任意:
  - `AI_PROVIDER`（例: `xai`）
  - `AI_BASE_URL`（例: `https://api.x.ai/v1`）
  - `AI_MODEL`（例: `grok-3-latest`）

Actions タブから `Automated Community Note Writer` を手動実行できます。
成果物は artifact (`writer-outputs`) として保存されます。

---

## 注意

- 初期は `test_mode=true` で検証してください。
- 外部 AI API の利用料金は利用者負担です。
- コミュニティノートの品質向上のため、`evaluate_note` の閾値運用（`claim_opinion_score`）を推奨します。
