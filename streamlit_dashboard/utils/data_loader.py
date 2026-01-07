"""
データローダー - ローカルCSVからデータを読み込む
"""
import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st


def get_data_path():
    """データディレクトリのパスを取得"""
    # streamlit_dashboard/utils/data_loader.py から見て ../../data/
    return Path(__file__).parent.parent.parent / "data"


@st.cache_data
def load_responses():
    """responses.csvを読み込む"""
    data_path = get_data_path()
    responses_path = data_path / "responses.csv"
    
    if not responses_path.exists():
        st.error(f"データファイルが見つかりません: {responses_path}")
        return None
    
    df = pd.read_csv(responses_path)
    return df


@st.cache_data
def load_sessions():
    """sessions.csvを読み込む"""
    data_path = get_data_path()
    sessions_path = data_path / "sessions.csv"
    
    if not sessions_path.exists():
        st.error(f"データファイルが見つかりません: {sessions_path}")
        return None
    
    df = pd.read_csv(sessions_path)
    return df


@st.cache_data
def load_analysis_data():
    """分析用にデータを整形して読み込む"""
    responses = load_responses()
    sessions = load_sessions()
    
    if responses is None or sessions is None:
        return None
    
    # 完了したセッションのみ抽出
    completed_sessions = sessions[sessions['current_phase'] == 'completed']['id'].tolist()
    df = responses[responses['session_id'].isin(completed_sessions)].copy()
    
    # ユーザーIDを付与（session_idを短縮してユーザー識別子として使用）
    session_to_user = {sid: f"User{i+1}" for i, sid in enumerate(df['session_id'].unique())}
    df['user_id'] = df['session_id'].map(session_to_user)
    
    return df


def calculate_baseline_predictions(df):
    """ベースライン予測を計算"""
    df = df.copy()
    
    # 全体平均
    mean_confidence = df['confidence'].mean()
    mean_challenge = df['challenge'].mean()
    mean_gap = df['human_gap'].mean()
    
    # ベースライン1: 全体平均
    df['baseline_mean_confidence'] = mean_confidence
    df['baseline_mean_challenge'] = mean_challenge
    df['baseline_mean_gap'] = mean_gap
    
    # ベースライン2: ランダム（1-7の一様分布）
    np.random.seed(42)
    df['baseline_random_confidence'] = np.random.randint(1, 8, size=len(df))
    df['baseline_random_challenge'] = np.random.randint(1, 8, size=len(df))
    df['baseline_random_gap'] = df['baseline_random_confidence'] - df['baseline_random_challenge']
    
    # ベースライン3: 中央値（4）
    df['baseline_median_confidence'] = 4
    df['baseline_median_challenge'] = 4
    df['baseline_median_gap'] = 0
    
    # ベースライン4: ユーザー平均（各ユーザーの平均値）
    user_means = df.groupby('user_id').agg({
        'confidence': 'mean',
        'challenge': 'mean',
        'human_gap': 'mean'
    }).reset_index()
    user_means.columns = ['user_id', 'user_mean_confidence', 'user_mean_challenge', 'user_mean_gap']
    df = df.merge(user_means, on='user_id', how='left')
    df['baseline_user_mean_confidence'] = df['user_mean_confidence']
    df['baseline_user_mean_challenge'] = df['user_mean_challenge']
    df['baseline_user_mean_gap'] = df['user_mean_gap']
    
    return df


def calculate_metrics(y_true, y_pred):
    """評価指標を計算"""
    from scipy import stats
    
    # 欠損値を除去
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = np.array(y_true)[mask]
    y_pred = np.array(y_pred)[mask]
    
    if len(y_true) == 0:
        return {}
    
    # MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # 相関係数
    if np.std(y_pred) > 0:
        corr, p_value = stats.pearsonr(y_true, y_pred)
    else:
        corr, p_value = 0, 1
    
    # R²（決定係数）
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # 完全一致率
    exact_match = np.mean(y_true == np.round(y_pred)) * 100
    
    # ±1以内の割合
    within_1 = np.mean(np.abs(y_true - y_pred) <= 1) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        '相関係数': corr,
        'p値': p_value,
        'R²': r2,
        '完全一致率(%)': exact_match,
        '±1以内(%)': within_1,
        'サンプル数': len(y_true)
    }


def get_summary_statistics(df):
    """基本統計量を計算"""
    stats_dict = {
        'サンプル数': len(df),
        'ユーザー数': df['user_id'].nunique() if 'user_id' in df.columns else df['session_id'].nunique(),
        '問題数': df['problem_id'].nunique(),
        '自信度平均': df['confidence'].mean(),
        '自信度標準偏差': df['confidence'].std(),
        '挑戦度平均': df['challenge'].mean(),
        '挑戦度標準偏差': df['challenge'].std(),
        'GAP平均': df['human_gap'].mean(),
        'GAP標準偏差': df['human_gap'].std(),
    }
    return stats_dict
