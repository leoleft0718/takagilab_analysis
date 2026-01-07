# 実験結果ダッシュボード

Streamlitで構築された実験データ分析ダッシュボードです。

## 📂 フォルダ構成

```
streamlit_dashboard/
├── app.py                      # メインアプリ
├── pages/
│   ├── 1_📋_実験概要.py         # セクション1: 実験設計と評価指標
│   ├── 2_📊_ベースライン比較.py   # セクション2: LLM vs ベースライン
│   ├── 3_📈_分布比較.py          # セクション3: 人間 vs AI分布
│   ├── 4_🔬_混合効果モデル.py    # セクション4: 線形混合モデル分析
│   ├── 5_🔗_自信度挑戦度相関.py   # セクション5: 相関分析
│   ├── 6_🎯_推薦評価.py          # セクション6: 推薦システム評価
│   └── 7_👤_ユーザー別分析.py    # セクション7: ユーザー別分析
├── utils/
│   ├── __init__.py
│   ├── data_loader.py          # データ読み込み関数
│   └── charts.py               # グラフ生成関数
└── README.md
```

## 🚀 起動方法

```bash
cd streamlit_dashboard
streamlit run app.py
```

## 📊 機能一覧

### 1. 実験概要
- 実験設計の説明
- 評価指標の定義
- データ概要

### 2. ベースライン比較
- LLM vs 各種ベースライン（全体平均、ランダム、中央値、ユーザー平均）
- 自信度/挑戦度/GAPごとの比較表
- 勝敗ヒートマップ

### 3. 分布の比較
- 人間評価とAI予測の分布比較
- 並列ヒストグラム
- 分散の比較

### 4. 線形混合モデル分析
- モデル式の表示
- 固定効果・ランダム効果の推定
- 分散成分の円グラフ
- R²比較
- 残差診断プロット

### 5. 自信度と挑戦度の関係
- 散布図（回帰線付き）
- 相関係数・R²の表示
- 人間 vs AI の相関比較

### 6. 推薦システム評価
- 混同行列
- 適合率/再現率/F1スコア/正解率
- ゲージチャート
- GAP分類ヒートマップ

### 7. ユーザー別分析
- ユーザー別F1スコア棒グラフ
- 詳細テーブル
- 個別ユーザーの深掘り分析

## 📁 データ要件

以下のCSVファイルが `../data/` ディレクトリに必要です：

- `responses.csv`: 回答データ
  - 必要カラム: session_id, problem_id, confidence, challenge, ai_predicted_confidence, ai_predicted_challenge, human_gap, ai_gap
  
- `sessions.csv`: セッションデータ
  - 必要カラム: id, current_phase

## 🔧 依存パッケージ

```
streamlit
pandas
numpy
plotly
scipy
statsmodels
```
