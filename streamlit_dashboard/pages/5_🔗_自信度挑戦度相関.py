"""
🔗 自信度と挑戦度の関係ページ
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_analysis_data

st.set_page_config(page_title="自信度と挑戦度の関係", page_icon="🔗", layout="wide")

st.title("🔗 自信度と挑戦度の関係")
st.markdown("自信度と挑戦度の相関関係を分析します。")
st.markdown("---")

df = load_analysis_data()

if df is not None:
    # データ選択
    data_source = st.radio(
        "分析対象",
        ['人間の評価', 'AIの予測', '両方比較'],
        horizontal=True
    )
    
    st.markdown("---")
    
    if data_source == '人間の評価':
        x_col, y_col = 'confidence', 'challenge'
        label = '人間'
    elif data_source == 'AIの予測':
        x_col, y_col = 'ai_predicted_confidence', 'ai_predicted_challenge'
        label = 'AI'
    else:
        x_col, y_col = 'confidence', 'challenge'
        label = '比較'
    
    # 相関係数の計算
    if data_source != '両方比較':
        corr, p_value = stats.pearsonr(df[x_col], df[y_col])
        r2 = corr ** 2
        
        st.header(f"📊 {label}の自信度 vs 挑戦度")
        
        # 散布図（回帰線付き）
        fig = px.scatter(
            df, x=x_col, y=y_col,
            trendline="ols",
            labels={x_col: '自信度', y_col: '挑戦度'},
            title=f'{label}の自信度 vs 挑戦度（r = {corr:.3f}）',
            opacity=0.6
        )
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        
        # メトリクス表示
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "相関係数 (r)", 
                f"{corr:.3f}",
                help="正: 正の相関、負: 負の相関"
            )
            if corr < -0.7:
                st.info("📉 強い負の相関")
            elif corr < -0.4:
                st.info("📉 中程度の負の相関")
            elif corr < -0.2:
                st.info("📉 弱い負の相関")
            elif corr < 0.2:
                st.info("➡️ ほぼ無相関")
            elif corr < 0.4:
                st.info("📈 弱い正の相関")
            elif corr < 0.7:
                st.info("📈 中程度の正の相関")
            else:
                st.info("📈 強い正の相関")
        
        with col2:
            st.metric(
                "決定係数 (R²)", 
                f"{r2*100:.1f}%",
                help="自信度で挑戦度の何%を説明できるか"
            )
        
        with col3:
            st.metric(
                "残差", 
                f"{(1-r2)*100:.1f}%",
                help="自信度では説明できない独自情報"
            )
            st.metric("p値", f"{p_value:.2e}")
        
        st.markdown("---")
        
        # 円グラフ（説明可能 vs 残差）
        st.header("📊 情報源の内訳")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                values=[r2 * 100, (1 - r2) * 100],
                names=['自信度で説明可能', '独自情報（残差）'],
                title='挑戦度の情報源',
                color_discrete_sequence=['#2ecc71', '#e74c3c']
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown(f"""
            ### 解釈
            
            **R² = {r2*100:.1f}%** ということは、
            
            - 挑戦度の **{r2*100:.1f}%** は自信度から予測可能
            - 残りの **{(1-r2)*100:.1f}%** は自信度とは独立した情報
            
            {'⚠️ **結論**: 自信度と挑戦度は強く相関しているが、完全には連動しないため、両方の質問項目が必要' if abs(corr) > 0.5 and r2 < 0.8 else ''}
            """)
        
        # 結論
        if abs(corr) > 0.5:
            if r2 < 0.8:
                st.warning(f"""
                ⚠️ **重要な発見**: 自信度と挑戦度は {abs(corr):.2f} の相関がありますが、
                R² = {r2*100:.1f}% であり、**{(1-r2)*100:.1f}%の独自情報**が存在します。
                
                → **両方の質問項目を維持することが推奨されます**
                """)
            else:
                st.info(f"""
                💡 自信度と挑戦度は非常に強く相関しています（r = {corr:.2f}, R² = {r2*100:.1f}%）。
                質問項目の統合を検討してもよいかもしれません。
                """)
    
    else:
        # 両方比較
        st.header("📊 人間 vs AI の相関比較")
        
        # 人間の相関
        corr_human, p_human = stats.pearsonr(df['confidence'], df['challenge'])
        r2_human = corr_human ** 2
        
        # AIの相関
        corr_ai, p_ai = stats.pearsonr(df['ai_predicted_confidence'], df['ai_predicted_challenge'])
        r2_ai = corr_ai ** 2
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("人間の評価")
            fig1 = px.scatter(
                df, x='confidence', y='challenge',
                trendline="ols",
                labels={'confidence': '自信度', 'challenge': '挑戦度'},
                title=f'人間: r = {corr_human:.3f}',
                opacity=0.6,
                color_discrete_sequence=['#636EFA']
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            st.metric("相関係数", f"{corr_human:.3f}")
            st.metric("R²", f"{r2_human*100:.1f}%")
        
        with col2:
            st.subheader("AIの予測")
            fig2 = px.scatter(
                df, x='ai_predicted_confidence', y='ai_predicted_challenge',
                trendline="ols",
                labels={'ai_predicted_confidence': '自信度', 'ai_predicted_challenge': '挑戦度'},
                title=f'AI: r = {corr_ai:.3f}',
                opacity=0.6,
                color_discrete_sequence=['#EF553B']
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            st.metric("相関係数", f"{corr_ai:.3f}")
            st.metric("R²", f"{r2_ai*100:.1f}%")
        
        st.markdown("---")
        
        # 比較まとめ
        st.header("📋 相関の比較まとめ")
        
        comparison_df = pd.DataFrame({
            '指標': ['相関係数 (r)', 'R²', 'p値'],
            '人間': [f"{corr_human:.3f}", f"{r2_human*100:.1f}%", f"{p_human:.2e}"],
            'AI': [f"{corr_ai:.3f}", f"{r2_ai*100:.1f}%", f"{p_ai:.2e}"]
        })
        
        st.table(comparison_df)
        
        # 差の解釈
        diff = abs(corr_human) - abs(corr_ai)
        if abs(diff) < 0.1:
            st.success("✅ 人間とAIの相関パターンはほぼ一致しています。AIは人間の評価パターンをよく再現しています。")
        elif diff > 0:
            st.info(f"📊 人間の方が自信度-挑戦度間の相関が強いです（差: {diff:.3f}）")
        else:
            st.info(f"📊 AIの方が自信度-挑戦度間の相関が強いです（差: {-diff:.3f}）")

else:
    st.error("データの読み込みに失敗しました。")
