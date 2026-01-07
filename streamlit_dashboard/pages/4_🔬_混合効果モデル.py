"""
🔬 線形混合モデル分析ページ
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_analysis_data, create_residual_plots

st.set_page_config(page_title="線形混合モデル分析", page_icon="🔬", layout="wide")

st.title("🔬 線形混合モデル分析")
st.markdown("ランダム効果を考慮した統計モデルによる分析結果です。")
st.markdown("---")

df = load_analysis_data()

if df is not None:
    # モデル式の表示
    st.header("📐 モデル式")
    
    st.latex(r"y_{ij} = \beta_0 + \beta_1 \cdot \text{ai\_gap}_{ij} + u_i + v_j + \varepsilon_{ij}")
    
    st.markdown("""
    **パラメータの説明:**
    - $y_{ij}$: ユーザー$i$の問題$j$に対する人間GAP
    - $\\beta_0$: 切片（固定効果）
    - $\\beta_1$: AI予測GAPの係数（固定効果）
    - $u_i$: ユーザーのランダム効果
    - $v_j$: 問題のランダム効果  
    - $\\varepsilon_{ij}$: 残差
    """)
    
    st.markdown("---")
    
    # 線形混合モデルの実行
    try:
        import statsmodels.formula.api as smf
        
        # モデルフィッティング
        model = smf.mixedlm(
            "human_gap ~ ai_gap",
            data=df,
            groups=df["user_id"],
            re_formula="1",
            vc_formula={"problem_id": "0 + C(problem_id)"}
        )
        
        with st.spinner("モデルをフィッティング中..."):
            try:
                result = model.fit(method='powell')
                model_fitted = True
            except:
                # 簡略化したモデル
                model = smf.mixedlm(
                    "human_gap ~ ai_gap",
                    data=df,
                    groups=df["user_id"]
                )
                result = model.fit()
                model_fitted = True
                st.info("簡略化したモデルを使用しています（問題ランダム効果なし）")
        
        if model_fitted:
            # 固定効果の表示
            st.header("📊 固定効果（Fixed Effects）")
            
            fixed_effects = pd.DataFrame({
                '係数': [result.fe_params['Intercept'], result.fe_params['ai_gap']],
                '標準誤差': [result.bse_fe['Intercept'], result.bse_fe['ai_gap']],
                'z値': [result.tvalues['Intercept'], result.tvalues['ai_gap']],
                'p値': [result.pvalues['Intercept'], result.pvalues['ai_gap']]
            }, index=['切片', 'AI予測GAP'])
            
            st.dataframe(
                fixed_effects.style.format({
                    '係数': '{:.4f}',
                    '標準誤差': '{:.4f}',
                    'z値': '{:.3f}',
                    'p値': '{:.4f}'
                }),
                use_container_width=True
            )
            
            # 係数の解釈
            coef = result.fe_params['ai_gap']
            if result.pvalues['ai_gap'] < 0.05:
                st.success(f"""
                ✅ **AI予測GAPは統計的に有意** (p < 0.05)
                
                AI予測GAPが1増加すると、人間GAPは平均 **{coef:.3f}** 増加します。
                """)
            else:
                st.warning("AI予測GAPは統計的に有意ではありません (p >= 0.05)")
            
            st.markdown("---")
            
            # 分散成分
            st.header("📈 分散成分")
            
            # ランダム効果の分散を取得
            re_var = result.cov_re.iloc[0, 0] if hasattr(result, 'cov_re') else 0
            resid_var = result.scale
            
            # 予測値から固定効果の分散を推定
            predicted = result.fittedvalues
            fe_var = np.var(result.fe_params['Intercept'] + result.fe_params['ai_gap'] * df['ai_gap'])
            
            total_var = re_var + resid_var + fe_var
            
            variance_components = {
                '固定効果（AI予測）': fe_var / total_var * 100,
                'ユーザー効果': re_var / total_var * 100,
                '残差': resid_var / total_var * 100
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 円グラフ
                fig = px.pie(
                    values=list(variance_components.values()),
                    names=list(variance_components.keys()),
                    title="分散成分の割合",
                    color_discrete_sequence=['#636EFA', '#EF553B', '#00CC96']
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # 詳細テーブル
                var_df = pd.DataFrame({
                    '成分': list(variance_components.keys()),
                    '割合(%)': list(variance_components.values())
                })
                st.dataframe(
                    var_df.style.format({'割合(%)': '{:.1f}'}),
                    use_container_width=True
                )
                
                st.info("""
                **解釈:**
                - 固定効果: AI予測GAPで説明できる変動
                - ユーザー効果: ユーザー間の個人差
                - 残差: モデルで説明できない変動
                """)
            
            st.markdown("---")
            
            # R²の比較
            st.header("📊 モデル適合度（R²）")
            
            # 周辺R²と条件付きR²の計算
            y_true = df['human_gap']
            y_pred = result.fittedvalues
            
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            
            r2_conditional = 1 - (ss_res / ss_tot)
            r2_marginal = variance_components['固定効果（AI予測）'] / 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "周辺R²（固定効果のみ）", 
                    f"{r2_marginal*100:.1f}%",
                    help="AI予測GAPのみで説明できる分散の割合"
                )
            
            with col2:
                st.metric(
                    "条件付きR²（全効果）", 
                    f"{r2_conditional*100:.1f}%",
                    help="固定効果とランダム効果を含めた説明力"
                )
            
            # 棒グラフ
            r2_df = pd.DataFrame({
                '種類': ['周辺R²\n(AI予測のみ)', '条件付きR²\n(全効果)'],
                'R²': [r2_marginal * 100, r2_conditional * 100]
            })
            
            fig = px.bar(
                r2_df, x='種類', y='R²',
                color='種類',
                title='決定係数（R²）の比較',
                text='R²'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(showlegend=False, yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # 残差診断
            st.header("🔍 残差診断")
            
            residuals = y_true - y_pred
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['残差ヒストグラム', '残差 vs 予測値', 'Q-Qプロット', '残差の箱ひげ図']
            )
            
            # 残差ヒストグラム
            fig.add_trace(
                go.Histogram(x=residuals, name='残差', marker_color='#636EFA', nbinsx=20),
                row=1, col=1
            )
            
            # 残差 vs 予測値
            fig.add_trace(
                go.Scatter(x=y_pred, y=residuals, mode='markers', name='残差', 
                          marker=dict(color='#636EFA', opacity=0.5)),
                row=1, col=2
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
            
            # Q-Qプロット
            sorted_residuals = np.sort(residuals)
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_residuals)))
            fig.add_trace(
                go.Scatter(x=theoretical_quantiles, y=sorted_residuals, mode='markers', 
                          name='Q-Q', marker=dict(color='#636EFA')),
                row=2, col=1
            )
            min_val = min(theoretical_quantiles.min(), sorted_residuals.min())
            max_val = max(theoretical_quantiles.max(), sorted_residuals.max())
            fig.add_trace(
                go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines',
                          line=dict(color='red', dash='dash'), showlegend=False),
                row=2, col=1
            )
            
            # 残差の箱ひげ図
            fig.add_trace(
                go.Box(y=residuals, name='残差', marker_color='#636EFA'),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
            
            # 正規性検定
            stat, p_value = stats.shapiro(residuals[:min(5000, len(residuals))])
            
            if p_value > 0.05:
                st.success(f"✅ 残差は正規分布に従っている可能性が高い (Shapiro-Wilk p = {p_value:.4f})")
            else:
                st.warning(f"⚠️ 残差は正規分布から逸脱している可能性 (Shapiro-Wilk p = {p_value:.4f})")
    
    except ImportError:
        st.error("statsmodelsがインストールされていません。`pip install statsmodels`を実行してください。")
    except Exception as e:
        st.error(f"モデルのフィッティング中にエラーが発生しました: {e}")
        
        # 代替として単純な線形回帰を表示
        st.header("📊 代替: 単純線形回帰分析")
        
        from scipy.stats import linregress
        
        slope, intercept, r_value, p_value, std_err = linregress(df['ai_gap'], df['human_gap'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("傾き (β₁)", f"{slope:.4f}")
        with col2:
            st.metric("切片 (β₀)", f"{intercept:.4f}")
        with col3:
            st.metric("R²", f"{r_value**2:.4f}")
        
        # 散布図と回帰線
        fig = px.scatter(df, x='ai_gap', y='human_gap', 
                        title='AI予測GAP vs 人間GAP',
                        labels={'ai_gap': 'AI予測GAP', 'human_gap': '人間GAP'},
                        trendline='ols')
        st.plotly_chart(fig, use_container_width=True)

else:
    st.error("データの読み込みに失敗しました。")
