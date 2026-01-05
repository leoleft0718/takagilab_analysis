# 📊 実験データ分析ダッシュボード

AIの予測と人間の回答を比較分析するStreamlitアプリケーション。

## 🎯 概要

本システムは、線形代数学習支援システムにおけるAI予測精度を評価・分析するためのダッシュボードです。

### 分析対象
- **人間のGAP**: 自信度 - 挑戦度（ユーザーが回答した値）
- **AI予測GAP**: AI予測自信度 - AI予測挑戦度（AIが予測した値）
- **GAP差**: |人間GAP - AI GAP|（予測精度の指標）

## 📈 主な分析機能

### 1. 概要統計
- 完了セッション数
- 総回答数（final_checkフェーズ）
- ピアソン相関係数（r）とp値
- 完全一致率（人間GAP = AI GAP）
- ±1以内、±2以内の一致率
- 平均人間GAP / 平均AI GAP
- 決定係数（R²）

### 2. 可視化（6タブ構成）

#### 散布図・ヒートマップ
- 人間GAP vs AI GAP 散布図（ジッター付き）
- GAPの組み合わせ頻度ヒートマップ
- 相関分析の詳細（MAE, RMSE）

#### 分布
- 人間GAPの分布
- AI予測GAPの分布
- GAP差の分布
- 自信度・挑戦度の比較ヒストグラム

#### 問題別分析
- GAP差が大きい問題 Top 10（AIの予測がずれやすい問題）
- GAP差が小さい問題 Top 10（AIの予測が正確な問題）
- 問題ごとの平均GAP比較
- 問題別統計テーブル
- 問題の詳細表示と個別分布

#### 知識要素別
- 知識要素別 平均GAP比較
- 知識要素別統計テーブル
- 知識要素別相関係数

#### セッション別
- セッション別 平均GAP散布図
- セッション別統計テーブル

#### 問題一覧
- 全問題の一覧表示
- 各問題の統計情報

### 3. フィルター機能
- 学年別フィルター
- 知識要素別フィルター
- データ再読み込み

### 4. データエクスポート
- 回答データCSV
- セッションデータCSV
- 問題データCSV

## 🚀 セットアップ

### ローカル環境

```bash
# リポジトリをクローン
git clone https://github.com/leoleft0718/takagilab_analysis.git
cd takagilab_analysis

# 仮想環境を作成・有効化
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存パッケージをインストール
pip install -r requirements.txt

# 環境変数を設定
cp .env.example .env
# .env ファイルに DATABASE_URL を設定

# アプリを起動
streamlit run app.py
```

### 環境変数

```env
DATABASE_URL="postgresql://user:password@host:port/database"
```

## 📦 依存パッケージ

- streamlit >= 1.28.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- psycopg2-binary >= 2.9.0
- plotly >= 5.18.0
- scipy >= 1.11.0
- python-dotenv >= 1.0.0
- sqlalchemy >= 2.0.0

## 🌐 デプロイ

Streamlit Community Cloud でホスティング：
https://share.streamlit.io

### Secrets設定
Streamlit Cloud の Settings → Secrets に以下を追加：
```toml
DATABASE_URL = "postgresql://..."
```

## 📊 データベース構造

### 使用テーブル
- `sessions`: セッション情報
- `users`: ユーザー情報（学年、専攻など）
- `responses`: 回答データ（自信度、挑戦度、AI予測値）
- `problems`: 問題情報（知識要素、問題文）

### 主要カラム
| カラム | 説明 |
|--------|------|
| confidence | 人間の自信度 (1-5) |
| challenge | 人間の挑戦度 (1-5) |
| ai_predicted_confidence | AI予測の自信度 |
| ai_predicted_challenge | AI予測の挑戦度 |
| knowledge_component | 知識要素 |

## 📝 更新履歴

- 2026-01-05: 初回リリース
  - UUID JSON シリアライズ対応
  - use_container_width 非推奨警告対応
  - 独立リポジトリとして分離
