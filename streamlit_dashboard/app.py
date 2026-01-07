"""
📊 実験結果ダッシュボード - メインアプリ
"""
import streamlit as st
import sys
from pathlib import Path

# utils パスを追加
sys.path.insert(0, str(Path(__file__).parent))

from utils import load_analysis_data, get_summary_statistics

# ページ設定
st.set_page_config(
    page_title="実験結果ダッシュボード",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ヘッダー
st.markdown('<p class="main-header">📊 実験結果ダッシュボード</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">LLMによる自己評価予測システムの分析結果</p>', unsafe_allow_html=True)

# サイドバー
st.sidebar.title("📊 ナビゲーション")
st.sidebar.info("""
**セクション一覧:**
1. 🏠 ホーム（このページ）
2. 📋 実験概要
3. 📊 ベースライン比較
4. 📈 分布の比較
5. 🔬 線形混合モデル分析
6. 🔗 自信度と挑戦度の関係
7. 🎯 推薦システム評価
8. 👤 ユーザー別分析
""")

# データ読み込み
df = load_analysis_data()

if df is not None:
    # 基本統計
    stats = get_summary_statistics(df)
    
    st.markdown("---")
    st.header("📈 データ概要")
    
    # メトリクス表示
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 サンプル数", f"{stats['サンプル数']:,}")
    with col2:
        st.metric("👥 ユーザー数", f"{stats['ユーザー数']}")
    with col3:
        st.metric("📝 問題数", f"{stats['問題数']}")
    with col4:
        st.metric("📅 分析日", "2026年1月7日")
    
    st.markdown("---")
    
    # 概要説明
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 研究目的")
        st.markdown("""
        本研究では、**LLM（大規模言語モデル）を用いて学習者の自己評価を予測**し、
        効果的な問題推薦システムの構築を目指しています。
        
        主な評価指標：
        - **自信度**: 問題を解ける自信（1-7スケール）
        - **挑戦度**: 問題に挑戦したい意欲（1-7スケール）  
        - **GAP**: 自信度 - 挑戦度（学習適性の指標）
        """)
    
    with col2:
        st.subheader("📊 分析内容")
        st.markdown("""
        このダッシュボードでは以下の分析結果を確認できます：
        
        1. **ベースライン比較**: LLM予測と各種ベースラインの性能比較
        2. **分布分析**: 人間評価とAI予測の分布比較
        3. **混合効果モデル**: 統計的モデリング結果
        4. **相関分析**: 自信度と挑戦度の関係性
        5. **推薦評価**: 問題推薦の精度評価
        6. **ユーザー分析**: 個人差の分析
        """)
    
    st.markdown("---")
    
    # データプレビュー
    with st.expander("📋 データプレビュー", expanded=False):
        st.dataframe(
            df[['user_id', 'problem_id', 'confidence', 'challenge', 
                'ai_predicted_confidence', 'ai_predicted_challenge', 
                'human_gap', 'ai_gap']].head(20),
            use_container_width=True
        )
    
    # クイック統計
    with st.expander("📊 基本統計量", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**自信度**")
            st.write(f"- 平均: {stats['自信度平均']:.2f}")
            st.write(f"- 標準偏差: {stats['自信度標準偏差']:.2f}")
        
        with col2:
            st.markdown("**挑戦度**")
            st.write(f"- 平均: {stats['挑戦度平均']:.2f}")
            st.write(f"- 標準偏差: {stats['挑戦度標準偏差']:.2f}")
        
        with col3:
            st.markdown("**GAP（自信度-挑戦度）**")
            st.write(f"- 平均: {stats['GAP平均']:.2f}")
            st.write(f"- 標準偏差: {stats['GAP標準偏差']:.2f}")

else:
    st.error("データの読み込みに失敗しました。dataフォルダにCSVファイルがあることを確認してください。")
    st.info("必要なファイル: data/responses.csv, data/sessions.csv")

# フッター
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    📊 実験結果ダッシュボード | Built with Streamlit
</div>
""", unsafe_allow_html=True)
