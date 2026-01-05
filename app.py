import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import os
from pathlib import Path
from sqlalchemy import create_engine
import warnings

# 警告を抑制
warnings.filterwarnings('ignore', category=UserWarning)

# ページ設定
st.set_page_config(
    page_title="実験データ分析",
    page_icon="📊",
    layout="wide"
)

def load_env():
    """環境変数を.envまたは.env.localから読み込む"""
    # まず同じディレクトリの.envを試す
    env_path = Path(__file__).parent / '.env'
    if not env_path.exists():
        # 親ディレクトリの.env.localを試す
        env_path = Path(__file__).parent.parent / '.env.local'
    
    env_vars = {}
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    value = value.strip().strip('"').strip("'")
                    env_vars[key.strip()] = value
    return env_vars

@st.cache_resource
def get_engine():
    """SQLAlchemyエンジンを取得"""
    env_vars = load_env()
    database_url = env_vars.get('DATABASE_URL') or os.getenv('DATABASE_URL')
    
    if not database_url:
        st.error("DATABASE_URLが設定されていません。")
        return None
    
    try:
        engine = create_engine(database_url)
        return engine
    except Exception as e:
        st.error(f"データベース接続エラー: {e}")
        return None

@st.cache_data(ttl=60)
def load_data():
    """データベースからデータを読み込む"""
    engine = get_engine()
    if not engine:
        return None, None, None
    
    try:
        sessions_query = """
            SELECT 
                s.id, s.user_id, s.started_at, s.ended_at, s.current_phase,
                u.grade, u.major, u.linear_algebra_status, u.confidence_rating
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.current_phase = 'completed'
            ORDER BY s.ended_at DESC
        """
        
        responses_query = """
            SELECT 
                r.id, r.session_id, r.problem_id, r.phase,
                r.confidence, r.challenge, r.free_text,
                r.ai_predicted_confidence, r.ai_predicted_challenge,
                r.created_at,
                p.knowledge_component, p.description_main, p.description_sub, p.answer
            FROM responses r
            JOIN problems p ON r.problem_id = p.id
            JOIN sessions s ON r.session_id = s.id
            WHERE s.current_phase = 'completed'
            AND r.phase = 'final_check'
            AND r.ai_predicted_confidence IS NOT NULL
            AND r.ai_predicted_challenge IS NOT NULL
            ORDER BY r.created_at ASC
        """
        
        problems_query = """
            SELECT id, knowledge_component, description_main, description_sub, answer
            FROM problems ORDER BY id
        """
        
        sessions_df = pd.read_sql(sessions_query, engine)
        responses_df = pd.read_sql(responses_query, engine)
        problems_df = pd.read_sql(problems_query, engine)
        
        # UUIDを文字列に変換
        for col in ['id', 'user_id', 'session_id', 'problem_id']:
            if col in sessions_df.columns:
                sessions_df[col] = sessions_df[col].astype(str)
            if col in responses_df.columns:
                responses_df[col] = responses_df[col].astype(str)
            if col in problems_df.columns:
                problems_df[col] = problems_df[col].astype(str)
        
        # GAP計算
        responses_df['human_gap'] = responses_df['confidence'] - responses_df['challenge']
        responses_df['ai_gap'] = responses_df['ai_predicted_confidence'] - responses_df['ai_predicted_challenge']
        responses_df['gap_difference'] = (responses_df['human_gap'] - responses_df['ai_gap']).abs()
        
        # ベースライン計算用
        # 全体平均GAP
        global_mean_gap = responses_df['human_gap'].mean()
        responses_df['global_mean_gap'] = global_mean_gap
        
        # 問題別平均GAP（Leave-One-Out方式: 自分を除外した平均）
        # 各行について、同じ問題の他のユーザーの平均を計算
        def calc_problem_loo_mean(row, df):
            """自分を除外した問題別平均を計算"""
            same_problem = df[(df['problem_id'] == row['problem_id']) & (df['session_id'] != row['session_id'])]
            if len(same_problem) > 0:
                return same_problem['human_gap'].mean()
            else:
                # 他のユーザーがいない場合は全体平均を使用
                return df[df['session_id'] != row['session_id']]['human_gap'].mean()
        
        responses_df['problem_mean_gap'] = responses_df.apply(
            lambda row: calc_problem_loo_mean(row, responses_df), axis=1
        )
        
        # ユーザー別平均GAP（Leave-One-Out方式: 予測対象の問題を除外）
        def calc_user_loo_mean(row, df):
            """予測対象の問題を除外したユーザー平均を計算"""
            other_problems = df[(df['session_id'] == row['session_id']) & (df['problem_id'] != row['problem_id'])]
            if len(other_problems) > 0:
                return other_problems['human_gap'].mean()
            else:
                # 他の問題がない場合は全体平均を使用
                return df[df['session_id'] != row['session_id']]['human_gap'].mean()
        
        responses_df['user_mean_gap'] = responses_df.apply(
            lambda row: calc_user_loo_mean(row, responses_df), axis=1
        )
        
        # 各ベースラインとの誤差
        responses_df['llm_error'] = responses_df['gap_difference']
        responses_df['global_error'] = (responses_df['human_gap'] - responses_df['global_mean_gap']).abs()
        responses_df['problem_error'] = (responses_df['human_gap'] - responses_df['problem_mean_gap']).abs()
        responses_df['user_error'] = (responses_df['human_gap'] - responses_df['user_mean_gap']).abs()
        
        return sessions_df, responses_df, problems_df
        
    except Exception as e:
        st.error(f"データ読み込みエラー: {e}")
        return None, None, None

def calculate_metrics(actual, predicted):
    """各種メトリクスを計算"""
    if len(actual) < 2:
        return {'r': 0, 'p_value': 1, 'r2': 0, 'mae': 0, 'rmse': 0}
    
    # 定数配列の場合は相関係数を計算できない
    if np.std(actual) == 0 or np.std(predicted) == 0:
        r, p_value = 0, 1
    else:
        r, p_value = stats.pearsonr(actual, predicted)
    
    mae = np.abs(actual - predicted).mean()
    rmse = np.sqrt(((actual - predicted) ** 2).mean())
    
    return {
        'r': r,
        'p_value': p_value,
        'r2': r ** 2,
        'mae': mae,
        'rmse': rmse
    }

def main():
    st.title("📊 GAP分析ダッシュボード")
    st.markdown("**GAP = 自信度 - 挑戦度** の分析（AIの予測と人間の回答の比較）")
    st.caption("💡 サイドバーから「自信度分析」「挑戦度分析」ページに移動できます")
    
    # データ読み込み
    with st.spinner("データを読み込み中..."):
        sessions_df, responses_df, problems_df = load_data()
    
    if sessions_df is None or responses_df is None:
        st.error("データの読み込みに失敗しました")
        st.stop()
    
    if len(responses_df) == 0:
        st.warning("分析対象のデータがありません")
        st.stop()
    
    # サイドバー
    st.sidebar.header("🔧 設定")
    if st.sidebar.button("🔄 データを再読み込み"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    
    # タブ構成
    tab1, tab2, tab3, tab4 = st.tabs(["📈 全体像", "📝 問題別", "👤 人別", "🎯 適合率・再現率"])
    
    # ========== タブ1: 全体像 ==========
    with tab1:
        render_overview_tab(responses_df, sessions_df)
    
    # ========== タブ2: 問題別 ==========
    with tab2:
        render_problem_tab(responses_df, problems_df)
    
    # ========== タブ3: 人別 ==========
    with tab3:
        render_user_tab(responses_df, sessions_df)
    
    # ========== タブ4: 適合率・再現率 ==========
    with tab4:
        render_precision_recall_tab(responses_df)


def render_overview_tab(df, sessions_df):
    """全体像タブの描画"""
    st.header("📈 全体像・概要")
    
    # === 基本統計（サマリーカード） ===
    st.subheader("基本統計")
    st.caption("📌 実験の規模を把握できます。データ収集の進捗確認や、分析の信頼性（サンプルサイズ）の判断に使用します。")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("完了セッション数", len(sessions_df))
    with col2:
        st.metric("総回答数", len(df))
    with col3:
        st.metric("問題数", df['problem_id'].nunique())
    with col4:
        st.metric("ユーザー数", df['session_id'].nunique())
    
    st.divider()
    
    # === GAP分布比較（ヒストグラム並列） ===
    st.subheader("GAP分布比較")
    st.caption("📌 人間とAIのGAP値（自信度−挑戦度）の分布を比較します。分布の形状が似ていればAIは人間の傾向をよく捉えています。人間GAPが特定の値に偏っている場合、AIがその偏りを再現できているかを確認できます。")
    col_hist1, col_hist2 = st.columns(2)
    
    with col_hist1:
        fig_human = px.histogram(df, x='human_gap', nbins=11, title='人間GAP分布',
                                  color_discrete_sequence=['#636EFA'])
        fig_human.update_layout(xaxis_title='GAP値', yaxis_title='件数', height=300)
        st.plotly_chart(fig_human, key="human_gap_hist")
    
    with col_hist2:
        fig_ai = px.histogram(df, x='ai_gap', nbins=11, title='AI予測GAP分布',
                              color_discrete_sequence=['#EF553B'])
        fig_ai.update_layout(xaxis_title='GAP値', yaxis_title='件数', height=300)
        st.plotly_chart(fig_ai, key="ai_gap_hist")
    
    st.divider()
    
    # === GAPの組み合わせ頻度ヒートマップ ===
    st.subheader("GAPの組み合わせ頻度ヒートマップ")
    st.caption("📌 人間GAPとAI GAPの組み合わせごとの回答数を可視化します。対角線上（左下から右上）に回答が集中していれば予測精度が高いことを示します。対角線から離れた位置に多い場合、AIの予測がずれていることを意味します。")
    
    # クロス集計
    heatmap_data = pd.crosstab(df['ai_gap'], df['human_gap'])
    # 欠けている値を埋める
    all_gaps = range(-5, 6)
    for gap in all_gaps:
        if gap not in heatmap_data.index:
            heatmap_data.loc[gap] = 0
        if gap not in heatmap_data.columns:
            heatmap_data[gap] = 0
    heatmap_data = heatmap_data.sort_index().sort_index(axis=1)
    
    fig_gap_heatmap = px.imshow(
        heatmap_data,
        labels=dict(x="人間GAP", y="AI GAP", color="回答数"),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        text_auto=True,
        color_continuous_scale='Blues',
        title="GAPの組み合わせ頻度ヒートマップ"
    )
    fig_gap_heatmap.update_layout(height=450)
    st.plotly_chart(fig_gap_heatmap, key="gap_combination_heatmap")
    
    st.divider()
    
    # === 予測精度（散布図 + 回帰線） ===
    st.subheader("予測精度（散布図）")
    st.caption("📌 人間GAPとAI予測GAPの関係を散布図で表示します。点が対角線（完全一致線）に近いほど予測精度が高いです。r（相関係数）は線形関係の強さ、R²は予測の説明力、MAE/RMSEは予測誤差の大きさを示します。")
    
    # 計算方法の説明（折りたたみ）
    with st.expander("📐 計算方法の詳細"):
        st.markdown("""
### 評価指標の計算方法

#### 相関係数 (Pearson's r)
$$r = \\frac{\\sum_{i=1}^{n}(x_i - \\bar{x})(y_i - \\bar{y})}{\\sqrt{\\sum_{i=1}^{n}(x_i - \\bar{x})^2} \\sqrt{\\sum_{i=1}^{n}(y_i - \\bar{y})^2}}$$

- $x_i$: 人間GAP、$y_i$: AI予測GAP
- 範囲: -1 〜 1（1に近いほど強い正の相関）
- **解釈**: 0.7以上で強い相関、0.4-0.7で中程度、0.4未満で弱い相関

#### 決定係数 (R²)
$$R^2 = r^2$$

- 範囲: 0 〜 1
- **解釈**: AI予測が人間GAPの変動をどれだけ説明できるかを示す（0.5なら50%を説明）

#### 平均絶対誤差 (MAE: Mean Absolute Error)
$$MAE = \\frac{1}{n}\\sum_{i=1}^{n}|y_i - \\hat{y}_i|$$

- $y_i$: 人間GAP（実測値）、$\\hat{y}_i$: AI予測GAP（予測値）
- **解釈**: 平均的な予測誤差の大きさ。単位はGAP値と同じ（例: MAE=1.5なら平均1.5ポイントずれている）

#### 二乗平均平方根誤差 (RMSE: Root Mean Squared Error)
$$RMSE = \\sqrt{\\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2}$$

- **解釈**: MAEと似ているが、大きな誤差をより重くペナルティする。外れ値に敏感。
        """)
    
    metrics = calculate_metrics(df['human_gap'].values, df['ai_gap'].values)
    
    col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
    with col_metrics1:
        st.metric("相関係数 (r)", f"{metrics['r']:.4f}")
    with col_metrics2:
        st.metric("決定係数 (R²)", f"{metrics['r2']:.4f}")
    with col_metrics3:
        st.metric("MAE", f"{metrics['mae']:.2f}")
    with col_metrics4:
        st.metric("RMSE", f"{metrics['rmse']:.2f}")
    
    # 散布図
    fig_scatter = px.scatter(df, x='human_gap', y='ai_gap', 
                             title='人間GAP vs AI予測GAP',
                             opacity=0.5)
    # 回帰線
    z = np.polyfit(df['human_gap'], df['ai_gap'], 1)
    x_line = np.linspace(df['human_gap'].min(), df['human_gap'].max(), 100)
    y_line = z[0] * x_line + z[1]
    fig_scatter.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', 
                                     name='回帰線', line=dict(color='red')))
    # 完全一致線
    fig_scatter.add_trace(go.Scatter(x=[-5, 5], y=[-5, 5], mode='lines',
                                     name='完全一致線', line=dict(dash='dash', color='gray')))
    fig_scatter.update_layout(height=400, xaxis_title='人間GAP', yaxis_title='AI GAP')
    st.plotly_chart(fig_scatter, key="scatter_gap")
    
    st.divider()
    
    # === 一致率 ===
    st.subheader("一致率")
    st.caption("📌 AI予測が人間の回答とどれだけ一致しているかを示します。完全一致は厳しい基準、±1以内・±2以内は実用的な許容範囲での精度です。教育支援では±1以内の精度があれば実用的とされます。")
    
    # 計算方法の説明（折りたたみ）
    with st.expander("📐 一致率の計算方法"):
        st.markdown("""
### 一致率の計算方法

#### 完全一致率
$$\\text{完全一致率} = \\frac{\\sum_{i=1}^{n} \\mathbb{1}[y_i = \\hat{y}_i]}{n} \\times 100\\%$$

- $\\mathbb{1}[\\cdot]$: 条件を満たすとき1、そうでないとき0
- 人間GAPとAI予測GAPが完全に一致した回答の割合

#### ±k以内の一致率
$$\\text{±k以内} = \\frac{\\sum_{i=1}^{n} \\mathbb{1}[|y_i - \\hat{y}_i| \\leq k]}{n} \\times 100\\%$$

- 人間GAPとAI予測GAPの差がk以内の回答の割合
- ±1以内: 1ポイント以内の誤差を許容
- ±2以内: 2ポイント以内の誤差を許容
        """)
    
    exact = (df['human_gap'] == df['ai_gap']).sum()
    within1 = (df['gap_difference'] <= 1).sum()
    within2 = (df['gap_difference'] <= 2).sum()
    n = len(df)
    
    col_match1, col_match2, col_match3 = st.columns(3)
    with col_match1:
        st.metric("完全一致", f"{exact/n*100:.1f}%", f"{exact}件")
    with col_match2:
        st.metric("±1以内", f"{within1/n*100:.1f}%", f"{within1}件")
    with col_match3:
        st.metric("±2以内", f"{within2/n*100:.1f}%", f"{within2}件")
    
    # 一致率の棒グラフ
    match_data = pd.DataFrame({
        '条件': ['完全一致', '±1以内', '±2以内'],
        '割合': [exact/n*100, within1/n*100, within2/n*100]
    })
    fig_match = px.bar(match_data, x='条件', y='割合', title='一致率',
                       color='条件', text='割合')
    fig_match.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_match.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig_match, key="match_rate")
    
    st.divider()
    
    # === GAP差の分布 ===
    st.subheader("GAP差（|人間GAP - AI GAP|）の分布")
    st.caption("📌 予測誤差の分布を確認します。0に近い値が多いほど精度が高いです。大きな誤差（外れ値）がどの程度あるかも重要で、外れ値が多い場合は特定の条件で予測が難しいことを示唆します。")
    fig_diff = px.histogram(df, x='gap_difference', nbins=10, title='GAP差の分布',
                            color_discrete_sequence=['#00CC96'])
    fig_diff.update_layout(xaxis_title='GAP差', yaxis_title='件数', height=300)
    st.plotly_chart(fig_diff, key="gap_diff_hist")
    
    st.divider()
    
    # === ベースライン総合比較 ===
    st.subheader("🆕 ベースライン総合比較")
    st.caption("📌 LLM予測の価値を検証します。単純なベースライン（平均値を予測として使う方法）と比較し、LLMがそれらを上回っているかを確認します。MAEが低いほど良い予測です。LLMが全てのベースラインを下回れば、AIによる個別予測の有効性が示されます。")
    
    # 計算方法の説明（折りたたみ）
    with st.expander("📐 ベースラインの計算方法"):
        st.markdown("""
### ベースラインの定義

⚠️ **Leave-One-Out方式を採用**: 予測対象のデータを除外して平均を計算しています。これにより、データリークを防ぎ、公正な比較が可能です。

#### 全体平均ベースライン
$$\\hat{y}_i = \\bar{y} = \\frac{1}{n}\\sum_{j=1}^{n} y_j$$

- 全回答の人間GAP平均値を、すべての予測値として使用
- 最もシンプルなベースライン

#### 問題平均ベースライン（Leave-One-Out）
$$\\hat{y}_{i,p} = \\frac{1}{n_p - 1}\\sum_{j \\in P, j \\neq i} y_j$$

- 問題 $p$ に対する **自分以外** の回答者の人間GAP平均値を予測値として使用
- 予測対象のユーザーを除外することで、公正な評価が可能
- 「この問題に対しては、他の回答者と同じようなGAPになるだろう」という仮定

#### ユーザー平均ベースライン（Leave-One-Out）
$$\\hat{y}_{i,u} = \\frac{1}{n_u - 1}\\sum_{j \\in U, j \\neq i} y_j$$

- ユーザー $u$ の **予測対象の問題以外** の人間GAP平均値を予測値として使用
- 予測対象の問題を除外することで、公正な評価が可能
- 「このユーザーは、他の問題と同じようなGAPになるだろう」という仮定

### 比較の意味
- **LLM > ベースライン**: LLMの予測がベースラインより悪い（MAEが高い）
- **LLM < ベースライン**: LLMの予測がベースラインより良い（MAEが低い）
- LLMが全ベースラインに勝っていれば、**個別予測の価値**が示される
        """)
    
    st.markdown("**ベースラインの説明:**")
    st.markdown("""
- **全体平均**: 全回答のGAP平均値を常に予測として使用
- **問題平均**: 各問題の「自分以外」のGAP平均値を予測として使用（Leave-One-Out）
- **ユーザー平均**: 各ユーザーの「この問題以外」のGAP平均値を予測として使用（Leave-One-Out）
    """)
    
    # 各ベースラインのメトリクス計算
    baselines = {
        'LLM予測': df['ai_gap'].values,
        '全体平均': np.full(len(df), df['human_gap'].mean()),
        '問題平均': df['problem_mean_gap'].values,
        'ユーザー平均': df['user_mean_gap'].values
    }
    
    baseline_results = []
    for name, pred in baselines.items():
        m = calculate_metrics(df['human_gap'].values, pred)
        exact_match = (df['human_gap'].values == np.round(pred)).sum() / len(df) * 100
        within1_match = (np.abs(df['human_gap'].values - pred) <= 1).sum() / len(df) * 100
        baseline_results.append({
            'ベースライン': name,
            'MAE': m['mae'],
            '相関係数 (r)': m['r'],
            'R²': m['r2'],
            '完全一致率 (%)': exact_match,
            '±1以内 (%)': within1_match
        })
    
    baseline_df = pd.DataFrame(baseline_results)
    st.dataframe(baseline_df.round(4), hide_index=True)
    
    # MAE比較の棒グラフ
    fig_baseline = px.bar(baseline_df, x='ベースライン', y='MAE', 
                          title='ベースライン別MAE比較（低いほど良い）',
                          color='ベースライン', text='MAE')
    fig_baseline.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_baseline.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_baseline, key="baseline_mae")


def render_problem_tab(df, problems_df):
    """問題別タブの描画"""
    st.header("📝 問題別分析")
    
    # === 問題×GAP ヒートマップ ===
    st.subheader("問題×GAP ヒートマップ")
    st.caption("📌 各問題でどのGAP値が多いかを可視化します。問題ごとの回答傾向の違いが分かります。特定の問題で極端なGAP値が集中している場合、その問題の難易度や性質に特徴があることを示唆します。")
    
    # 問題IDを短縮
    df['problem_short'] = df['problem_id'].str[:8]
    
    col_hm1, col_hm2 = st.columns(2)
    with col_hm1:
        heatmap_human = df.groupby(['problem_short', 'human_gap']).size().unstack(fill_value=0)
        fig_hm_human = px.imshow(heatmap_human, title='人間GAP × 問題', 
                                  labels=dict(x='GAP値', y='問題ID', color='件数'),
                                  color_continuous_scale='Blues')
        fig_hm_human.update_layout(height=400)
        st.plotly_chart(fig_hm_human, key="hm_human_problem")
    
    with col_hm2:
        heatmap_ai = df.groupby(['problem_short', 'ai_gap']).size().unstack(fill_value=0)
        fig_hm_ai = px.imshow(heatmap_ai, title='AI GAP × 問題',
                              labels=dict(x='GAP値', y='問題ID', color='件数'),
                              color_continuous_scale='Reds')
        fig_hm_ai.update_layout(height=400)
        st.plotly_chart(fig_hm_ai, key="hm_ai_problem")
    
    st.divider()
    
    # === 問題別MAE（棒グラフ） ===
    st.subheader("問題別MAE（LLM予測）")
    st.caption("📌 各問題に対するLLMの予測誤差を示します。MAEが高い問題はAIにとって予測が難しい問題です。これらの問題を分析することで、AIの弱点や改善ポイントを特定できます。")
    problem_stats = df.groupby('problem_id').agg({
        'llm_error': 'mean',
        'problem_error': 'mean',
        'confidence': 'mean',
        'challenge': 'mean',
        'human_gap': ['mean', 'std', 'count'],
        'ai_gap': 'mean'
    }).reset_index()
    problem_stats.columns = ['problem_id', 'llm_mae', 'problem_baseline_mae', 
                             'avg_confidence', 'avg_challenge', 
                             'avg_human_gap', 'std_human_gap', 'count', 'avg_ai_gap']
    problem_stats['problem_short'] = problem_stats['problem_id'].str[:8]
    problem_stats = problem_stats.sort_values('llm_mae', ascending=False)
    
    fig_problem_mae = px.bar(problem_stats, x='problem_short', y='llm_mae',
                              title='問題別MAE（降順）', 
                              color='llm_mae', color_continuous_scale='Reds')
    fig_problem_mae.update_layout(height=350, xaxis_title='問題ID', yaxis_title='MAE')
    st.plotly_chart(fig_problem_mae, key="problem_mae")
    
    st.divider()
    
    # === 問題特性一覧（データテーブル） ===
    st.subheader("問題特性一覧")
    st.caption("📌 各問題の統計情報を一覧表示します。平均自信度・挑戦度から問題の難易度感を、人間GAPとAI GAPの差からAIの予測バイアスを把握できます。")
    display_cols = ['problem_short', 'count', 'avg_confidence', 'avg_challenge', 
                    'avg_human_gap', 'avg_ai_gap', 'llm_mae']
    display_df = problem_stats[display_cols].copy()
    display_df.columns = ['問題ID', '回答数', '平均自信度', '平均挑戦度', 
                          '平均人間GAP', '平均AI GAP', 'LLM MAE']
    st.dataframe(display_df.round(2), hide_index=True)
    
    st.divider()
    
    # === 予測誤差の傾向（箱ひげ図） ===
    st.subheader("予測誤差の傾向（箱ひげ図）")
    st.caption("📌 各問題の予測誤差のばらつきを可視化します。箱が大きい（ばらつきが大きい）問題は、回答者によって反応が異なるため予測が難しい問題です。外れ値が多い問題も要注意です。")
    df['problem_short_box'] = df['problem_id'].str[:8]
    fig_box = px.box(df, x='problem_short_box', y='gap_difference', 
                     title='問題別GAP差のばらつき')
    fig_box.update_layout(height=400, xaxis_title='問題ID', yaxis_title='GAP差')
    st.plotly_chart(fig_box, key="box_problem")
    
    st.divider()
    
    # === 問題平均ベースライン比較 ===
    st.subheader("🆕 問題平均ベースライン比較")
    st.caption("📌 各問題で「その問題の過去の平均GAP」を予測として使う方法とLLMを比較します。LLMが勝っている問題では、AIが個人差を考慮した予測ができていることを意味します。負けている問題では、単純に平均を使った方が良い結果になっています。")
    st.markdown("**見方**: 散布図で対角線より下の点＝LLMが勝っている問題")
    
    # 並列棒グラフ
    comparison_df = problem_stats[['problem_short', 'llm_mae', 'problem_baseline_mae']].melt(
        id_vars='problem_short', var_name='手法', value_name='MAE'
    )
    comparison_df['手法'] = comparison_df['手法'].map({
        'llm_mae': 'LLM予測', 
        'problem_baseline_mae': '問題平均'
    })
    
    fig_compare = px.bar(comparison_df, x='problem_short', y='MAE', color='手法',
                         barmode='group', title='LLM vs 問題平均（MAE比較）')
    fig_compare.update_layout(height=400, xaxis_title='問題ID')
    st.plotly_chart(fig_compare, key="problem_compare")
    
    # 散布図
    col_sc1, col_sc2 = st.columns([2, 1])
    with col_sc1:
        fig_scatter_prob = px.scatter(problem_stats, x='problem_baseline_mae', y='llm_mae',
                                       hover_data=['problem_short', 'count'],
                                       title='問題平均MAE vs LLM MAE（対角線より下なら勝ち）')
        max_val = max(problem_stats['problem_baseline_mae'].max(), problem_stats['llm_mae'].max())
        fig_scatter_prob.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines',
                                              name='同等ライン', line=dict(dash='dash', color='gray')))
        fig_scatter_prob.update_layout(height=350, xaxis_title='問題平均MAE', yaxis_title='LLM MAE')
        st.plotly_chart(fig_scatter_prob, key="scatter_problem_baseline")
    
    # 勝敗サマリー
    with col_sc2:
        llm_wins = (problem_stats['llm_mae'] < problem_stats['problem_baseline_mae']).sum()
        baseline_wins = (problem_stats['llm_mae'] > problem_stats['problem_baseline_mae']).sum()
        ties = (problem_stats['llm_mae'] == problem_stats['problem_baseline_mae']).sum()
        
        st.markdown("### 勝敗サマリー")
        st.metric("LLM勝利", f"{llm_wins}問", help="LLMのMAEが問題平均より小さい")
        st.metric("問題平均勝利", f"{baseline_wins}問", help="問題平均のMAEがLLMより小さい")
        st.metric("引き分け", f"{ties}問")
        
        # パイチャート
        pie_data = pd.DataFrame({
            '結果': ['LLM勝利', '問題平均勝利', '引き分け'],
            '問題数': [llm_wins, baseline_wins, ties]
        })
        fig_pie = px.pie(pie_data, values='問題数', names='結果', title='勝敗割合')
        fig_pie.update_layout(height=250)
        st.plotly_chart(fig_pie, key="pie_problem")


def render_user_tab(df, sessions_df):
    """人別タブの描画"""
    st.header("👤 人別（セッション別）分析")
    
    # セッションIDを短縮
    df['session_short'] = df['session_id'].str[:8]
    
    # === ユーザー×GAP ヒートマップ ===
    st.subheader("ユーザー×GAP ヒートマップ")
    st.caption("📌 各ユーザーのGAP回答パターンを可視化します。ユーザーごとの回答傾向の違い（常に高いGAPを付ける人、バランスよく付ける人など）を把握できます。")
    
    col_hm1, col_hm2 = st.columns(2)
    with col_hm1:
        heatmap_human = df.groupby(['session_short', 'human_gap']).size().unstack(fill_value=0)
        fig_hm_human = px.imshow(heatmap_human, title='人間GAP × ユーザー',
                                  labels=dict(x='GAP値', y='ユーザーID', color='件数'),
                                  color_continuous_scale='Blues')
        fig_hm_human.update_layout(height=400)
        st.plotly_chart(fig_hm_human, key="hm_human_user")
    
    with col_hm2:
        heatmap_ai = df.groupby(['session_short', 'ai_gap']).size().unstack(fill_value=0)
        fig_hm_ai = px.imshow(heatmap_ai, title='AI GAP × ユーザー',
                              labels=dict(x='GAP値', y='ユーザーID', color='件数'),
                              color_continuous_scale='Reds')
        fig_hm_ai.update_layout(height=400)
        st.plotly_chart(fig_hm_ai, key="hm_ai_user")
    
    st.divider()
    
    # === ユーザー別MAE ===
    st.subheader("ユーザー別MAE（LLM予測）")
    st.caption("📌 各ユーザーに対するLLMの予測誤差を示します。MAEが高いユーザーはAIにとって予測が難しい人です。極端な回答をする人や、一貫性のない回答をする人は予測が難しい傾向があります。")
    user_stats = df.groupby('session_id').agg({
        'llm_error': 'mean',
        'user_error': 'mean',
        'confidence': 'mean',
        'challenge': 'mean',
        'human_gap': ['mean', 'std', 'count'],
        'ai_gap': 'mean'
    }).reset_index()
    user_stats.columns = ['session_id', 'llm_mae', 'user_baseline_mae',
                          'avg_confidence', 'avg_challenge',
                          'avg_human_gap', 'std_human_gap', 'count', 'avg_ai_gap']
    user_stats['session_short'] = user_stats['session_id'].str[:8]
    user_stats = user_stats.sort_values('llm_mae', ascending=False)
    
    fig_user_mae = px.bar(user_stats, x='session_short', y='llm_mae',
                          title='ユーザー別MAE（降順）',
                          color='llm_mae', color_continuous_scale='Reds')
    fig_user_mae.update_layout(height=350, xaxis_title='ユーザーID', yaxis_title='MAE')
    st.plotly_chart(fig_user_mae, key="user_mae")
    
    st.divider()
    
    # === 個人特性の可視化（表） ===
    st.subheader("個人特性一覧")
    st.caption("📌 各ユーザーの回答傾向を一覧表示します。平均自信度・挑戦度からユーザーの自己評価傾向を、LLM MAEから予測難易度を把握できます。")
    display_cols = ['session_short', 'count', 'avg_confidence', 'avg_challenge',
                    'avg_human_gap', 'avg_ai_gap', 'llm_mae']
    display_df = user_stats[display_cols].copy()
    display_df.columns = ['ユーザーID', '回答数', '平均自信度', '平均挑戦度',
                          '平均人間GAP', '平均AI GAP', 'LLM MAE']
    st.dataframe(display_df.round(2), hide_index=True)
    
    st.divider()
    
    # === ユーザー平均ベースライン比較 ===
    st.subheader("🆕 ユーザー平均ベースライン比較")
    st.caption("📌 各ユーザーで「そのユーザーの過去の平均GAP」を予測として使う方法とLLMを比較します。LLMが勝っているユーザーでは、AIが問題ごとの違いを考慮した予測ができていることを意味します。負けているユーザーでは、その人は一貫した回答パターンを持っており、平均で十分予測できることを示唆します。")
    st.markdown("**見方**: 散布図で対角線より下の点＝LLMが勝っているユーザー")
    
    # 並列棒グラフ
    comparison_df = user_stats[['session_short', 'llm_mae', 'user_baseline_mae']].melt(
        id_vars='session_short', var_name='手法', value_name='MAE'
    )
    comparison_df['手法'] = comparison_df['手法'].map({
        'llm_mae': 'LLM予測',
        'user_baseline_mae': 'ユーザー平均'
    })
    
    fig_compare = px.bar(comparison_df, x='session_short', y='MAE', color='手法',
                         barmode='group', title='LLM vs ユーザー平均（MAE比較）')
    fig_compare.update_layout(height=400, xaxis_title='ユーザーID')
    st.plotly_chart(fig_compare, key="user_compare")
    
    # 散布図
    col_sc1, col_sc2 = st.columns([2, 1])
    with col_sc1:
        fig_scatter_user = px.scatter(user_stats, x='user_baseline_mae', y='llm_mae',
                                       hover_data=['session_short', 'count'],
                                       title='ユーザー平均MAE vs LLM MAE（対角線より下なら勝ち）')
        max_val = max(user_stats['user_baseline_mae'].max(), user_stats['llm_mae'].max())
        fig_scatter_user.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines',
                                              name='同等ライン', line=dict(dash='dash', color='gray')))
        fig_scatter_user.update_layout(height=350, xaxis_title='ユーザー平均MAE', yaxis_title='LLM MAE')
        st.plotly_chart(fig_scatter_user, key="scatter_user_baseline")
    
    # 勝敗サマリー
    with col_sc2:
        llm_wins = (user_stats['llm_mae'] < user_stats['user_baseline_mae']).sum()
        baseline_wins = (user_stats['llm_mae'] > user_stats['user_baseline_mae']).sum()
        ties = (user_stats['llm_mae'] == user_stats['user_baseline_mae']).sum()
        
        st.markdown("### 勝敗サマリー")
        st.metric("LLM勝利", f"{llm_wins}人", help="LLMのMAEがユーザー平均より小さい")
        st.metric("ユーザー平均勝利", f"{baseline_wins}人", help="ユーザー平均のMAEがLLMより小さい")
        st.metric("引き分け", f"{ties}人")
        
        # パイチャート
        pie_data = pd.DataFrame({
            '結果': ['LLM勝利', 'ユーザー平均勝利', '引き分け'],
            'ユーザー数': [llm_wins, baseline_wins, ties]
        })
        fig_pie = px.pie(pie_data, values='ユーザー数', names='結果', title='勝敗割合')
        fig_pie.update_layout(height=250)
        st.plotly_chart(fig_pie, key="pie_user")


def render_precision_recall_tab(df):
    """適合率・再現率タブの描画"""
    st.header("🎯 適合率・再現率分析")
    st.markdown("AIの予測を分類問題として評価します。「GAP値が一致する/しない」を予測できているかを、適合率・再現率で分析します。")
    
    # 計算方法の説明（折りたたみ）
    with st.expander("📐 適合率・再現率の計算方法"):
        st.markdown("""
### 分類指標の定義

GAP値の一致判定を二値分類問題として捉えます。

#### 混同行列の要素
| | 人間: 一致 | 人間: 不一致 |
|---|---|---|
| **AI: 一致予測** | TP (True Positive) | FP (False Positive) |
| **AI: 不一致予測** | FN (False Negative) | TN (True Negative) |

#### 適合率 (Precision)
$$\\text{Precision} = \\frac{TP}{TP + FP}$$

**意味**: AIが「GAPが一致する」と予測したもののうち、実際に一致していた割合
- 高い → AIが「一致」と言ったら信頼できる
- 低い → AIが「一致」と言っても実際は不一致が多い

#### 再現率 (Recall)
$$\\text{Recall} = \\frac{TP}{TP + FN}$$

**意味**: 実際にGAPが一致していたもののうち、AIが「一致」と予測できた割合
- 高い → 一致するケースを見逃さない
- 低い → 一致するケースを見逃している

#### F1スコア
$$F1 = 2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}$$

**意味**: 適合率と再現率の調和平均。両方のバランスを見る指標。
        """)
    
    st.divider()
    
    # === GAP値ごとの分析 ===
    st.subheader("GAP値ごとの適合率・再現率")
    st.caption("📌 各GAP値（-5〜5）について、AIがそのGAP値を予測したときの精度を評価します。")
    
    gap_metrics = []
    for gap_value in range(-5, 6):
        # 人間がそのGAP値だった回答
        human_positive = (df['human_gap'] == gap_value)
        # AIがそのGAP値と予測した回答
        ai_positive = (df['ai_gap'] == gap_value)
        
        tp = (human_positive & ai_positive).sum()
        fp = (~human_positive & ai_positive).sum()
        fn = (human_positive & ~ai_positive).sum()
        tn = (~human_positive & ~ai_positive).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        gap_metrics.append({
            'GAP値': gap_value,
            '人間の実数': human_positive.sum(),
            'AI予測数': ai_positive.sum(),
            'TP': tp,
            'FP': fp,
            'FN': fn,
            '適合率': precision,
            '再現率': recall,
            'F1': f1
        })
    
    gap_metrics_df = pd.DataFrame(gap_metrics)
    
    # メトリクスの可視化
    col_pr1, col_pr2 = st.columns(2)
    
    with col_pr1:
        fig_precision = px.bar(gap_metrics_df, x='GAP値', y='適合率',
                               title='GAP値ごとの適合率',
                               color='適合率', color_continuous_scale='Blues')
        fig_precision.update_layout(height=350, yaxis_range=[0, 1])
        st.plotly_chart(fig_precision, key="precision_by_gap")
    
    with col_pr2:
        fig_recall = px.bar(gap_metrics_df, x='GAP値', y='再現率',
                            title='GAP値ごとの再現率',
                            color='再現率', color_continuous_scale='Greens')
        fig_recall.update_layout(height=350, yaxis_range=[0, 1])
        st.plotly_chart(fig_recall, key="recall_by_gap")
    
    # F1スコア
    fig_f1 = px.bar(gap_metrics_df, x='GAP値', y='F1',
                    title='GAP値ごとのF1スコア',
                    color='F1', color_continuous_scale='Purples')
    fig_f1.update_layout(height=300, yaxis_range=[0, 1])
    st.plotly_chart(fig_f1, key="f1_by_gap")
    
    # 詳細テーブル
    st.markdown("### GAP値ごとの詳細データ")
    display_df = gap_metrics_df.copy()
    display_df['適合率'] = display_df['適合率'].apply(lambda x: f"{x:.1%}")
    display_df['再現率'] = display_df['再現率'].apply(lambda x: f"{x:.1%}")
    display_df['F1'] = display_df['F1'].apply(lambda x: f"{x:.3f}")
    st.dataframe(display_df, hide_index=True)
    
    st.divider()
    
    # === 許容範囲別の分析 ===
    st.subheader("許容範囲別の適合率・再現率")
    st.caption("📌 「GAP差がk以内なら一致」とみなした場合の評価です。実用的な観点での精度を確認できます。")
    
    tolerance_metrics = []
    for tolerance in [0, 1, 2]:
        label = "完全一致" if tolerance == 0 else f"±{tolerance}以内"
        
        # 人間とAIの差がtolerance以内かどうか
        actual_match = (df['gap_difference'] <= tolerance)
        
        # AIの予測について：AIが「一致する」と予測したかどうかをどう定義するか
        # ここでは、AIの予測GAPと人間のGAPが近いかどうかで判断
        # → AI視点：AIが予測したGAPが、何らかの基準で「自信を持っている」かどうか
        
        # 別のアプローチ：各GAP値について、AIがそのGAP値を予測した場合を「一致予測」とする
        # 実際の一致：人間GAPとAI GAPの差がtolerance以内
        
        tp = actual_match.sum()  # 実際に一致していた数
        total = len(df)
        
        # AIの視点での分析：AIがある予測をしたとき、それが正しかった割合
        # ここでは「AIの予測が人間のGAPとtolerance以内」を成功とする
        
        tolerance_metrics.append({
            '許容範囲': label,
            '一致数': tp,
            '総数': total,
            '一致率': tp / total
        })
    
    tolerance_df = pd.DataFrame(tolerance_metrics)
    
    col_tol1, col_tol2 = st.columns([1, 2])
    with col_tol1:
        st.dataframe(tolerance_df, hide_index=True)
    with col_tol2:
        fig_tol = px.bar(tolerance_df, x='許容範囲', y='一致率',
                         title='許容範囲別の一致率',
                         color='一致率', color_continuous_scale='Viridis',
                         text='一致率')
        fig_tol.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig_tol.update_layout(height=300, yaxis_range=[0, 1])
        st.plotly_chart(fig_tol, key="tolerance_match")
    
    st.divider()
    
    # === AIがGAP=0と予測した場合の分析 ===
    st.subheader("AIがGAP=0と予測した場合の分析")
    st.caption("📌 AIが「自信と難易度が一致している（GAP=0）」と予測した場合、実際の人間のGAPがどうだったかを分析します。")
    
    # AIがGAP=0と予測した場合を抽出
    ai_gap_zero = df[df['ai_gap'] == 0]
    total_gap_zero = len(ai_gap_zero)
    
    if total_gap_zero > 0:
        # 人間のGAPが±1以内だった数
        human_within_1 = ai_gap_zero[abs(ai_gap_zero['human_gap']) <= 1]
        within_1_count = len(human_within_1)
        within_1_pct = within_1_count / total_gap_zero * 100
        
        # 完全一致数
        exact_match = len(ai_gap_zero[ai_gap_zero['human_gap'] == 0])
        exact_match_pct = exact_match / total_gap_zero * 100
        
        # メトリクス表示
        col_g0_1, col_g0_2, col_g0_3 = st.columns(3)
        with col_g0_1:
            st.metric("AIがGAP=0と予測した数", f"{total_gap_zero}件")
        with col_g0_2:
            st.metric("人間GAP=0（完全一致）", f"{exact_match}件", f"{exact_match_pct:.1f}%")
        with col_g0_3:
            st.metric("人間GAP±1以内", f"{within_1_count}件", f"{within_1_pct:.1f}%")
        
        # 人間GAP値の分布
        col_g0_chart, col_g0_table = st.columns([2, 1])
        
        with col_g0_chart:
            gap_dist = ai_gap_zero['human_gap'].value_counts().sort_index().reset_index()
            gap_dist.columns = ['人間GAP', '件数']
            gap_dist['±1以内'] = gap_dist['人間GAP'].apply(lambda x: '±1以内' if abs(x) <= 1 else '±2以上')
            
            fig_gap_dist = px.bar(gap_dist, x='人間GAP', y='件数',
                                  color='±1以内',
                                  color_discrete_map={'±1以内': '#2ecc71', '±2以上': '#e74c3c'},
                                  title='AIがGAP=0と予測した時の人間GAP分布')
            fig_gap_dist.update_layout(height=350, xaxis=dict(dtick=1))
            st.plotly_chart(fig_gap_dist, key="ai_gap_zero_dist")
        
        with col_g0_table:
            st.markdown("#### 詳細内訳")
            detail_data = []
            for gap in range(-5, 6):
                count = len(ai_gap_zero[ai_gap_zero['human_gap'] == gap])
                pct = count / total_gap_zero * 100
                mark = "✓" if abs(gap) <= 1 else ""
                detail_data.append({
                    '人間GAP': f"{gap:+d}",
                    '件数': count,
                    '割合': f"{pct:.1f}%",
                    '±1': mark
                })
            detail_df = pd.DataFrame(detail_data)
            st.dataframe(detail_df, hide_index=True, height=400)
        
        # 解釈
        with st.expander("📖 この分析の解釈"):
            st.markdown(f"""
### 結果の解釈

AIが「GAP=0（自信と難易度が一致）」と予測した **{total_gap_zero}件** のうち：
- **完全一致（人間GAP=0）**: {exact_match}件 ({exact_match_pct:.1f}%)
- **許容範囲内（±1以内）**: {within_1_count}件 ({within_1_pct:.1f}%)

#### 意味
- AIがGAP=0と予測した場合、約 **{within_1_pct:.0f}%** の確率で人間のGAPも近い値（±1以内）でした
- 残りの **{100-within_1_pct:.0f}%** はAIの予測と人間の実際のGAPに±2以上の乖離がありました

#### 傾向分析
""")
            # 過信・過小評価の傾向
            under_confident = len(ai_gap_zero[ai_gap_zero['human_gap'] < -1])
            over_confident = len(ai_gap_zero[ai_gap_zero['human_gap'] > 1])
            st.markdown(f"""
- **人間が過小評価（GAP≤-2）**: {under_confident}件 ({under_confident/total_gap_zero*100:.1f}%) - AIは一致と予測したが、人間は難しいと感じていた
- **人間が過信（GAP≥2）**: {over_confident}件 ({over_confident/total_gap_zero*100:.1f}%) - AIは一致と予測したが、人間は自信過剰だった
            """)
    else:
        st.warning("AIがGAP=0と予測したデータがありません。")
    
    st.divider()
    
    # === 混同行列（完全一致） ===
    st.subheader("混同行列")
    st.caption("📌 AIの予測と人間の回答の組み合わせを可視化します。対角線上の値が大きいほど予測精度が高いです。")
    
    # GAP値の混同行列（簡略化版：-5〜5を3カテゴリに）
    def categorize_gap(gap):
        if gap <= -2:
            return '低 (≤-2)'
        elif gap >= 2:
            return '高 (≥2)'
        else:
            return '中 (-1〜1)'
    
    df['human_gap_cat'] = df['human_gap'].apply(categorize_gap)
    df['ai_gap_cat'] = df['ai_gap'].apply(categorize_gap)
    
    # カテゴリ順序を指定
    cat_order = ['低 (≤-2)', '中 (-1〜1)', '高 (≥2)']
    
    confusion_matrix = pd.crosstab(
        pd.Categorical(df['ai_gap_cat'], categories=cat_order, ordered=True),
        pd.Categorical(df['human_gap_cat'], categories=cat_order, ordered=True),
        dropna=False
    )
    
    fig_cm = px.imshow(confusion_matrix,
                       labels=dict(x="人間GAP（実際）", y="AI GAP（予測）", color="件数"),
                       x=cat_order,
                       y=cat_order,
                       text_auto=True,
                       color_continuous_scale='Blues',
                       title='GAP値カテゴリの混同行列')
    fig_cm.update_layout(height=400)
    st.plotly_chart(fig_cm, key="confusion_matrix_cat")
    
    # 完全なGAP値の混同行列
    with st.expander("📊 詳細な混同行列（全GAP値）"):
        full_cm = pd.crosstab(df['ai_gap'], df['human_gap'])
        # 欠けている値を埋める
        all_gaps = range(-5, 6)
        for gap in all_gaps:
            if gap not in full_cm.index:
                full_cm.loc[gap] = 0
            if gap not in full_cm.columns:
                full_cm[gap] = 0
        full_cm = full_cm.sort_index().sort_index(axis=1)
        
        fig_full_cm = px.imshow(full_cm,
                                labels=dict(x="人間GAP", y="AI GAP", color="件数"),
                                x=full_cm.columns.tolist(),
                                y=full_cm.index.tolist(),
                                text_auto=True,
                                color_continuous_scale='Blues',
                                title='全GAP値の混同行列')
        fig_full_cm.update_layout(height=500)
        st.plotly_chart(fig_full_cm, key="confusion_matrix_full")
    
    st.divider()
    
    # === 総合評価 ===
    st.subheader("総合評価サマリー")
    
    # マクロ平均の計算
    macro_precision = gap_metrics_df['適合率'].mean()
    macro_recall = gap_metrics_df['再現率'].mean()
    macro_f1 = gap_metrics_df['F1'].mean()
    
    # 重み付け平均（人間の実数で重み付け）
    total_human = gap_metrics_df['人間の実数'].sum()
    weighted_precision = (gap_metrics_df['適合率'] * gap_metrics_df['人間の実数']).sum() / total_human
    weighted_recall = (gap_metrics_df['再現率'] * gap_metrics_df['人間の実数']).sum() / total_human
    weighted_f1 = (gap_metrics_df['F1'] * gap_metrics_df['人間の実数']).sum() / total_human
    
    col_sum1, col_sum2 = st.columns(2)
    
    with col_sum1:
        st.markdown("#### マクロ平均（各GAP値を均等に評価）")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("適合率", f"{macro_precision:.1%}")
        with col_m2:
            st.metric("再現率", f"{macro_recall:.1%}")
        with col_m3:
            st.metric("F1スコア", f"{macro_f1:.3f}")
    
    with col_sum2:
        st.markdown("#### 重み付け平均（出現頻度で重み付け）")
        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            st.metric("適合率", f"{weighted_precision:.1%}")
        with col_w2:
            st.metric("再現率", f"{weighted_recall:.1%}")
        with col_w3:
            st.metric("F1スコア", f"{weighted_f1:.3f}")
    
    # 解釈の説明
    with st.expander("📖 結果の解釈"):
        st.markdown("""
### 結果の読み方

#### 適合率が高いGAP値
AIがそのGAP値を予測したとき、信頼性が高い。教育支援では、AIの予測を信じて良い。

#### 再現率が高いGAP値
実際にそのGAP値だったケースを、AIがよく捉えている。見逃しが少ない。

#### 適合率が低いGAP値
AIがそのGAP値を予測しても、実際は違うことが多い。過信に注意。

#### 再現率が低いGAP値
実際にそのGAP値だったケースを、AIが見逃している。補完的な確認が必要。

#### マクロ平均 vs 重み付け平均
- **マクロ平均**: 全GAP値を均等に評価。少数派のGAP値も同じ重みで評価される。
- **重み付け平均**: 出現頻度で重み付け。実際のデータ分布を反映した評価。
        """)


if __name__ == "__main__":
    main()
