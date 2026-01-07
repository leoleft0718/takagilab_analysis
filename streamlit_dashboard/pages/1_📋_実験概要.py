"""
📋 実験概要ページ
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_analysis_data, get_summary_statistics

st.set_page_config(page_title="実験概要", page_icon="📋", layout="wide")

st.title("📋 実験概要")
st.markdown("---")

# 実験設計の説明
st.header("🔬 実験設計")

col1, col2 = st.columns(2)

with col1:
    st.subheader("実験の流れ")
    st.markdown("""
    1. **参加者募集**: 大学で線形代数を学習中の学生
    2. **事前調査**: 学習状況・自信度の確認
    3. **問題提示**: 線形代数の問題を順次提示
    4. **自己評価収集**: 各問題に対する自信度・挑戦度を回答
    5. **LLM予測**: GPT-4による自己評価の予測
    """)

with col2:
    st.subheader("評価スケール")
    st.markdown("""
    **自信度（Confidence）**: 1-7スケール
    - 1: 全く自信がない
    - 4: どちらとも言えない
    - 7: 非常に自信がある
    
    **挑戦度（Challenge）**: 1-7スケール
    - 1: 全く挑戦したくない
    - 4: どちらとも言えない
    - 7: 非常に挑戦したい
    """)

st.markdown("---")

# 評価指標の説明
st.header("📊 評価指標の定義")

with st.expander("評価指標の詳細", expanded=True):
    metrics_df = pd.DataFrame({
        '指標': ['MAE', 'RMSE', '相関係数 (r)', 'R²', '完全一致率', '±1以内'],
        '定義': [
            '平均絶対誤差: |予測値 - 実測値| の平均',
            '二乗平均平方根誤差: √(Σ(予測値 - 実測値)² / n)',
            'ピアソン相関係数: 予測値と実測値の線形関係の強さ',
            '決定係数: 予測モデルが説明できる分散の割合',
            '予測値と実測値が完全に一致する割合',
            '予測値と実測値の差が±1以内に収まる割合'
        ],
        '理想値': ['0 (低いほど良い)', '0 (低いほど良い)', '1 (高いほど良い)', 
                  '1 (高いほど良い)', '100% (高いほど良い)', '100% (高いほど良い)']
    })
    st.table(metrics_df)

st.markdown("---")

# GAP指標の説明
st.header("🎯 GAP指標について")

col1, col2 = st.columns(2)

with col1:
    st.latex(r"\text{GAP} = \text{自信度} - \text{挑戦度}")
    
    st.markdown("""
    **GAPの解釈:**
    - **GAP > 0**: 自信度 > 挑戦度 → 問題が簡単すぎる可能性
    - **GAP = 0**: 自信度 = 挑戦度 → 適切な難易度
    - **GAP < 0**: 自信度 < 挑戦度 → 問題が難しすぎる可能性
    """)

with col2:
    st.info("""
    **学習適性の観点から:**
    
    GAPが0に近い問題は、学習者にとって「適度に挑戦的」な問題であり、
    最も効果的な学習が期待できます。
    
    このGAPを正確に予測することで、個々の学習者に最適な問題を推薦できます。
    """)

st.markdown("---")

# データ概要
st.header("📈 データ概要")

df = load_analysis_data()

if df is not None:
    stats = get_summary_statistics(df)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("総サンプル数", f"{stats['サンプル数']:,}")
        st.metric("ユーザー数", stats['ユーザー数'])
    
    with col2:
        st.metric("問題数", stats['問題数'])
        st.metric("自信度平均", f"{stats['自信度平均']:.2f}")
    
    with col3:
        st.metric("挑戦度平均", f"{stats['挑戦度平均']:.2f}")
        st.metric("GAP平均", f"{stats['GAP平均']:.2f}")
    
    # 知識コンポーネント別の分布
    if 'knowledge_component' in df.columns:
        st.subheader("📚 知識コンポーネント別サンプル数")
        kc_counts = df['knowledge_component'].value_counts()
        st.bar_chart(kc_counts)
else:
    st.error("データの読み込みに失敗しました。")
