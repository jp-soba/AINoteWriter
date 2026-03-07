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
  - `--feed-lang` 未指定時は英語のみ（API デフォルト）
  - `--feed-lang ja` で日本語、`--feed-lang all` で全言語
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
- AI 実行環境（下書きを生成する場合）
  - xAI API を使う場合: `AI_API_KEY`
  - Claude Code プランを使う場合: VPS 上で Claude Code/Agent SDK が利用できる状態

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

## AI プロバイダー設定（xAI / Claude Agent SDK）

### xAI API（従来どおり）

```dotenv
AI_PROVIDER=xai
AI_API_KEY=...
AI_BASE_URL=https://api.x.ai/v1
AI_MODEL=grok-3-latest
```

### Claude Agent SDK（Claude Code プラン）

```dotenv
AI_PROVIDER=claude_agent
AI_MODEL=claude-sonnet-4-5
AI_TIMEOUT_SEC=120
CLAUDE_CLI_PATH=claude
CLAUDE_MAX_TURNS=4
CLAUDE_USE_CLI_FALLBACK=true
```

- まず Python の `claude-code-sdk` を使って実行します。
- SDK が使えない場合、`CLAUDE_USE_CLI_FALLBACK=true` なら `claude --print` に自動フォールバックします。
- Claude 経路では `AI_API_KEY` は不要です（Claude Code 側の認証状態を利用）。

### VPS での Claude 利用準備

1. VPS で Claude Code の認証を完了
2. `claude` コマンドが実行できることを確認
3. 本プロジェクトの `.env` を `AI_PROVIDER=claude_agent` に設定
4. `ai-note-writer run ...` で通常どおり実行

---

## CLI 実行

### 1) 英語フィード（デフォルト）で下書きのみ
```bash
ai-note-writer run --num-posts 5 --test-mode true --submit-notes false --evaluate-before-submit true
```

### 2) 日本語フィードで投稿あり
```bash
ai-note-writer run --num-posts 5 --test-mode true --submit-notes true --evaluate-before-submit true --feed-lang ja
```

### 3) 全言語フィード
```bash
ai-note-writer run --num-posts 10 --test-mode true --submit-notes true --feed-lang all
```

### 4) 自分が書いたノートの確認
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
  - `CLAUDE_CLI_PATH`（例: `claude`）
  - `CLAUDE_MAX_TURNS`（例: `4`）
  - `CLAUDE_USE_CLI_FALLBACK`（例: `true`）

Actions タブから `Automated Community Note Writer` を手動実行できます。
成果物は artifact (`writer-outputs`) として保存されます。

デフォルトでは英語フィードを使用します。手動実行時に `feed_lang` を `ja` や `all` に設定することで言語を変更できます。

---

## 注意

- 初期は `test_mode=true` で検証してください。
- 外部 AI API の利用料金は利用者負担です。
- コミュニティノートの品質向上のため、`evaluate_note` の閾値運用（`claim_opinion_score`）を推奨します。
