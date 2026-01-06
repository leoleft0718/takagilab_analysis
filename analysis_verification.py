import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr, shapiro, mannwhitneyu, ttest_ind
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Hiragino Sans'
import sys
import os

def calculate_metrics(actual, predicted):
    """評価指標を計算"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # 相関係数
    r, p_value = pearsonr(actual, predicted)
    r2 = r ** 2
    
    # MAE
    mae = np.abs(actual - predicted).mean()
    
    # 一致率
    exact_match = (actual == predicted).sum() / len(actual) * 100
    within_1 = (np.abs(actual - predicted) <= 1).sum() / len(actual) * 100
    within_2 = (np.abs(actual - predicted) <= 2).sum() / len(actual) * 100
    
    return {
        'r': r,
        'p_value': p_value,
        'r2': r2,
        'mae': mae,
        'exact_match': exact_match,
        'within_1': within_1,
        'within_2': within_2
    }

def print_metrics(name, metrics):
    """評価指標を表示"""
    print(f'\n{name}:')
    print(f'  相関係数 (r): {metrics["r"]:.3f}')
    print(f'  決定係数 (R²): {metrics["r2"]:.3f}')
    print(f'  MAE: {metrics["mae"]:.2f}')
    print(f'  完全一致率: {metrics["exact_match"]:.1f}%')
    print(f'  ±1以内: {metrics["within_1"]:.1f}%')

def calc_loo_baselines(df, value_col):
    """Leave-One-Out方式でベースラインを計算"""
    result_df = df.copy()
    
    # 全体平均（自分を除く）
    global_mean = df[value_col].mean()
    result_df['global_mean'] = global_mean
    
    # 問題平均（LOO）
    def calc_problem_loo(row):
        same_problem = df[(df['problem_id'] == row['problem_id']) & (df['session_id'] != row['session_id'])]
        if len(same_problem) > 0:
            return same_problem[value_col].mean()
        else:
            return df[df['session_id'] != row['session_id']][value_col].mean()
    
    result_df['problem_mean'] = df.apply(calc_problem_loo, axis=1)
    
    # ユーザー平均（LOO）
    def calc_user_loo(row):
        other_problems = df[(df['session_id'] == row['session_id']) & (df['problem_id'] != row['problem_id'])]
        if len(other_problems) > 0:
            return other_problems[value_col].mean()
        else:
            return df[df['session_id'] != row['session_id']][value_col].mean()
    
    result_df['user_mean'] = df.apply(calc_user_loo, axis=1)
    
    return result_df

def main():
    print("処理を開始します...")

    # 1. データ読み込み
    try:
        sessions_df = pd.read_csv('data/sessions.csv')
        df = pd.read_csv('data/responses.csv')
        print(f'読み込み完了: {len(df)}件の回答データ (phase=final_check のみ)')
    except FileNotFoundError as e:
        print(f"エラー: データファイルが見つかりません。{e}")
        return

    # 2. 基本統計
    print('=' * 50)
    print('📊 基本統計')
    print('=' * 50)
    print(f'完了セッション数: {len(sessions_df)}')
    print(f'総回答数: {len(df)}')
    print(f'問題数: {df["problem_id"].nunique()}')
    print(f'ユーザー数: {df["session_id"].nunique()}')
    print('=' * 50)

    # 3. 評価関数の定義 (関数の定義は上部で行済)

    # 4. ベースライン計算関数の定義 (関数の定義は上部で行済)
    print('ベースライン計算関数を定義しました')

    # 5. 自信度予測の検証
    print('=' * 50)
    print('📊 自信度予測の検証')
    print('=' * 50)

    # 自信度のベースライン計算
    conf_df = calc_loo_baselines(df, 'confidence')

    # LLM予測
    llm_conf_metrics = calculate_metrics(conf_df['confidence'], conf_df['ai_predicted_confidence'])
    print_metrics('LLM予測', llm_conf_metrics)

    # 全体平均
    global_conf_metrics = calculate_metrics(conf_df['confidence'], conf_df['global_mean'])
    print_metrics('全体平均', global_conf_metrics)

    # 問題平均
    problem_conf_metrics = calculate_metrics(conf_df['confidence'], conf_df['problem_mean'])
    print_metrics('問題平均', problem_conf_metrics)

    # ユーザー平均
    user_conf_metrics = calculate_metrics(conf_df['confidence'], conf_df['user_mean'])
    print_metrics('ユーザー平均', user_conf_metrics)

    # 6. 挑戦度予測の検証
    print('=' * 50)
    print('📊 挑戦度予測の検証')
    print('=' * 50)

    # 挑戦度のベースライン計算
    chal_df = calc_loo_baselines(df, 'challenge')

    # LLM予測
    llm_chal_metrics = calculate_metrics(chal_df['challenge'], chal_df['ai_predicted_challenge'])
    print_metrics('LLM予測', llm_chal_metrics)

    # 全体平均
    global_chal_metrics = calculate_metrics(chal_df['challenge'], chal_df['global_mean'])
    print_metrics('全体平均', global_chal_metrics)

    # 問題平均
    problem_chal_metrics = calculate_metrics(chal_df['challenge'], chal_df['problem_mean'])
    print_metrics('問題平均', problem_chal_metrics)

    # ユーザー平均
    user_chal_metrics = calculate_metrics(chal_df['challenge'], chal_df['user_mean'])
    print_metrics('ユーザー平均', user_chal_metrics)

    # 7. GAP予測の検証
    print('=' * 50)
    print('📊 GAP予測の検証')
    print('=' * 50)

    # GAPのベースライン計算
    gap_df = calc_loo_baselines(df, 'human_gap')

    # LLM予測
    llm_gap_metrics = calculate_metrics(gap_df['human_gap'], gap_df['ai_gap'])
    print_metrics('LLM予測', llm_gap_metrics)

    # 全体平均
    global_gap_metrics = calculate_metrics(gap_df['human_gap'], gap_df['global_mean'])
    print_metrics('全体平均', global_gap_metrics)

    # 問題平均
    problem_gap_metrics = calculate_metrics(gap_df['human_gap'], gap_df['problem_mean'])
    print_metrics('問題平均', problem_gap_metrics)

    # ユーザー平均
    user_gap_metrics = calculate_metrics(gap_df['human_gap'], gap_df['user_mean'])
    print_metrics('ユーザー平均', user_gap_metrics)

    # 8. 混合効果モデル分析
    print('=' * 50)
    print('📊 混合効果モデル分析')
    print('=' * 50)

    # データ準備
    model_df = df[['human_gap', 'ai_gap', 'problem_id', 'session_id']].copy()
    model_df = model_df.rename(columns={'session_id': 'user_id'})
    model_df = model_df.dropna()

    # Full Model
    full_model = smf.mixedlm(
        'human_gap ~ ai_gap', 
        data=model_df, 
        groups=model_df['user_id'],
        re_formula='~1',
        vc_formula={'problem_id': '0 + C(problem_id)'}
    )
    full_result = full_model.fit()

    # 分散成分の抽出
    sigma2_user = full_result.cov_re.iloc[0, 0]
    sigma2_problem = full_result.vcomp[0] if len(full_result.vcomp) > 0 else 0
    sigma2_residual = full_result.scale
    total_variance = sigma2_problem + sigma2_user + sigma2_residual

    # ICC
    icc_problem = sigma2_problem / total_variance
    icc_user = sigma2_user / total_variance
    icc_residual = sigma2_residual / total_variance

    print(f'\n分散成分:')
    print(f'  問題効果 (ICC): {icc_problem:.1%}')
    print(f'  ユーザー効果 (ICC): {icc_user:.1%}')
    print(f'  残差: {icc_residual:.1%}')

    # 固定効果
    beta_ai = full_result.params['ai_gap']
    p_ai = full_result.pvalues['ai_gap']

    print(f'\n固定効果:')
    print(f'  LLM係数 (β): {beta_ai:.3f}')
    print(f'  p値: {p_ai:.4g}')

    # R² (Nakagawa & Schielzeth, 2013)
    fixed_pred = full_result.params['Intercept'] + full_result.params['ai_gap'] * model_df['ai_gap']
    var_fixed = fixed_pred.var()
    var_random = sigma2_problem + sigma2_user
    denom = var_fixed + var_random + sigma2_residual

    marginal_r2 = var_fixed / denom
    conditional_r2 = (var_fixed + var_random) / denom

    print(f'\nR²:')
    print(f'  周辺R² (固定効果のみ): {marginal_r2:.1%}')
    print(f'  条件付きR² (全効果): {conditional_r2:.1%}')

    # 9. 残差診断
    print('=' * 50)
    print('📊 残差診断')
    print('=' * 50)

    # 残差の計算
    residuals = model_df['human_gap'] - full_result.fittedvalues

    print(f'\n残差の統計量:')
    print(f'  平均: {residuals.mean():.4f}')
    print(f'  標準偏差: {residuals.std():.4f}')
    print(f'  歪度: {stats.skew(residuals):.4f}')
    print(f'  尖度: {stats.kurtosis(residuals):.4f}')

    # Shapiro-Wilk検定
    shapiro_stat, shapiro_p = shapiro(residuals)
    print(f'\nShapiro-Wilk検定:')
    print(f'  W = {shapiro_stat:.4f}')
    print(f'  p = {shapiro_p:.4f}')

    # 10. ピアソン vs スピアマン相関の比較
    print('=' * 50)
    print('📊 ピアソン vs スピアマン相関の比較')
    print('=' * 50)

    comparisons = [
        ('自信度', 'confidence', 'ai_predicted_confidence'),
        ('挑戦度', 'challenge', 'ai_predicted_challenge'),
        ('GAP', 'human_gap', 'ai_gap'),
    ]

    print(f'\n{"指標":<10} {"ピアソン(r)":<12} {"スピアマン(ρ)":<14} {"差":<8} {"判定"}')
    print('-' * 55)

    for name, human_col, ai_col in comparisons:
        pearson_r, _ = pearsonr(df[human_col], df[ai_col])
        spearman_r, _ = spearmanr(df[human_col], df[ai_col])
        diff = abs(pearson_r - spearman_r)
        judge = '✅ 一致' if diff < 0.05 else '⚠️ 乖離'
        print(f'{name:<10} {pearson_r:<12.3f} {spearman_r:<14.3f} {diff:<8.3f} {judge}')

    # 11. 人間の自信度と挑戦度の相関
    print('=' * 50)
    print('📊 人間の自信度と挑戦度の相関')
    print('=' * 50)

    conf_chal_r, conf_chal_p = pearsonr(df['confidence'], df['challenge'])
    print(f'\nPearson相関係数: r = {conf_chal_r:.4f}')
    print(f'p値: {conf_chal_p:.4g}')
    print(f'決定係数 (R²): {conf_chal_r**2:.4f}')

    if abs(conf_chal_r) >= 0.7:
        strength = '強い'
    elif abs(conf_chal_r) >= 0.4:
        strength = '中程度の'
    else:
        strength = '弱い'
    direction = '正の' if conf_chal_r > 0 else '負の'

    print(f'\n解釈: {strength}{direction}相関')

    # 12. 結果サマリー
    print('='*60)
    print('📋 考察.mdとの比較サマリー')
    print('='*60)

    print('\n【基本統計】')
    print(f'  セッション数: {len(sessions_df)} (期待値: 11)')
    print(f'  回答数: {len(df)} (期待値: 300)')
    print(f'  問題数: {df["problem_id"].nunique()} (期待値: 35)')
    print(f'  ユーザー数: {df["session_id"].nunique()} (期待値: 10)')

    print('\n【自信度予測 - LLM】')
    print(f'  MAE: {llm_conf_metrics["mae"]:.2f} (期待値: 1.09)')
    print(f'  r: {llm_conf_metrics["r"]:.3f} (期待値: 0.600)')
    print(f'  完全一致率: {llm_conf_metrics["exact_match"]:.1f}% (期待値: 32.7%)')

    print('\n【挑戦度予測 - LLM】')
    print(f'  MAE: {llm_chal_metrics["mae"]:.2f} (期待値: 1.23)')
    print(f'  r: {llm_chal_metrics["r"]:.3f} (期待値: 0.371)')

    print('\n【GAP予測 - LLM】')
    print(f'  MAE: {llm_gap_metrics["mae"]:.2f} (期待値: 2.11)')
    print(f'  r: {llm_gap_metrics["r"]:.3f} (期待値: 0.532)')

    print('\n【混合効果モデル】')
    print(f'  問題効果 (ICC): {icc_problem:.1%} (期待値: 27.1%)')
    print(f'  ユーザー効果 (ICC): {icc_user:.1%} (期待値: 10.2%)')
    print(f'  LLM係数 (β): {beta_ai:.3f} (期待値: 0.524)')
    print(f'  周辺R²: {marginal_r2:.1%} (期待値: 10.0%)')

    # 13. 推薦システムの検証: AI GAP = 0 の効果
    print('=' * 60)
    print('📊 推薦システムの検証: AI GAP = 0 の効果')
    print('=' * 60)

    # AI GAPが0のグループとそれ以外のグループを分ける
    df_ai_gap_zero = df[df['ai_gap'] == 0]
    df_ai_gap_nonzero = df[df['ai_gap'] != 0]

    print(f'\n【データの分布】')
    print(f'  AI GAP = 0 の回答数: {len(df_ai_gap_zero)} ({len(df_ai_gap_zero)/len(df)*100:.1f}%)')
    print(f'  AI GAP ≠ 0 の回答数: {len(df_ai_gap_nonzero)} ({len(df_ai_gap_nonzero)/len(df)*100:.1f}%)')

    # Human GAPの統計量を比較
    print(f'\n【Human GAP の統計量】')
    print(f'\n  AI GAP = 0 のグループ:')
    print(f'    平均: {df_ai_gap_zero["human_gap"].mean():.3f}')
    print(f'    標準偏差: {df_ai_gap_zero["human_gap"].std():.3f}')
    print(f'    中央値: {df_ai_gap_zero["human_gap"].median():.3f}')
    print(f'    |GAP|の平均: {df_ai_gap_zero["human_gap"].abs().mean():.3f}')

    print(f'\n  AI GAP ≠ 0 のグループ:')
    print(f'    平均: {df_ai_gap_nonzero["human_gap"].mean():.3f}')
    print(f'    標準偏差: {df_ai_gap_nonzero["human_gap"].std():.3f}')
    print(f'    中央値: {df_ai_gap_nonzero["human_gap"].median():.3f}')
    print(f'    |GAP|の平均: {df_ai_gap_nonzero["human_gap"].abs().mean():.3f}')

    # 統計的検定
    print('\n【統計的検定】')

    # 1. |Human GAP| の比較 (Mann-Whitney U検定 - ノンパラメトリック)
    abs_gap_zero = df_ai_gap_zero['human_gap'].abs()
    abs_gap_nonzero = df_ai_gap_nonzero['human_gap'].abs()

    # Mann-Whitney U検定
    u_stat, u_p = mannwhitneyu(abs_gap_zero, abs_gap_nonzero, alternative='less')
    print(f'\n  Mann-Whitney U検定 (|Human GAP|が小さいか):')
    print(f'    U統計量: {u_stat:.2f}')
    print(f'    p値 (片側): {u_p:.4f}')
    print(f'    判定: {"✅ 有意 (p < 0.05)" if u_p < 0.05 else "❌ 有意でない"}')

    # 2. Welchのt検定
    t_stat, t_p = ttest_ind(abs_gap_zero, abs_gap_nonzero, equal_var=False)
    print(f'\n  Welchのt検定 (|Human GAP|の平均比較):')
    print(f'    t統計量: {t_stat:.2f}')
    print(f'    p値 (両側): {t_p:.4f}')
    print(f'    判定: {"✅ 有意 (p < 0.05)" if t_p < 0.05 else "❌ 有意でない"}')

    # 効果量 (Cohen's d)
    mean_diff = abs_gap_zero.mean() - abs_gap_nonzero.mean()
    pooled_std = np.sqrt((abs_gap_zero.std()**2 + abs_gap_nonzero.std()**2) / 2)
    cohens_d = mean_diff / pooled_std
    print(f'\n  効果量 (Cohen\'s d): {cohens_d:.3f}')
    if abs(cohens_d) < 0.2:
        effect_size = '効果なし'
    elif abs(cohens_d) < 0.5:
        effect_size = '小'
    elif abs(cohens_d) < 0.8:
        effect_size = '中'
    else:
        effect_size = '大'
    print(f'    解釈: {effect_size}')

    # AI GAPの値別にHuman GAPの|絶対値|を分析
    print('\n【AI GAP値別の Human |GAP| 分析】')
    print('-' * 50)

    # AI GAPの各値に対して集計
    ai_gap_analysis = df.groupby('ai_gap').agg({
        'human_gap': ['count', 'mean', 'std', lambda x: x.abs().mean()]
    }).round(3)
    ai_gap_analysis.columns = ['件数', 'Human GAP平均', 'Human GAP標準偏差', '|Human GAP|平均']
    ai_gap_analysis = ai_gap_analysis.sort_index()

    print(ai_gap_analysis.to_string())

    # Human GAP = 0 の割合を計算
    print('\n【AI GAP値別の Human GAP = 0 の割合】')
    print('-' * 50)
    for ai_gap_val in sorted(df['ai_gap'].unique()):
        subset = df[df['ai_gap'] == ai_gap_val]
        human_gap_zero_rate = (subset['human_gap'] == 0).sum() / len(subset) * 100
        human_gap_small_rate = (subset['human_gap'].abs() <= 1).sum() / len(subset) * 100
        print(f'  AI GAP = {ai_gap_val:>2}: Human GAP=0 {human_gap_zero_rate:>5.1f}%, |Human GAP|≤1 {human_gap_small_rate:>5.1f}% (n={len(subset)})')

    # 結論
    print('\n' + '=' * 60)
    print('📋 結論: AI GAP = 0 の推薦効果')
    print('=' * 60)

    print(f'''
【検証結果】
  AI GAP = 0 の場合:
    - |Human GAP|の平均: {abs_gap_zero.mean():.3f}
    - Human GAP = 0 の割合: {(df_ai_gap_zero["human_gap"] == 0).mean()*100:.1f}%
    
  AI GAP ≠ 0 の場合:
    - |Human GAP|の平均: {abs_gap_nonzero.mean():.3f}
    - Human GAP = 0 の割合: {(df_ai_gap_nonzero["human_gap"] == 0).mean()*100:.1f}%

【統計的検定の結果】
  Mann-Whitney U検定 p値: {u_p:.4f}
  効果量 (Cohen's d): {cohens_d:.3f} ({effect_size})

【解釈】
''')

    if u_p < 0.05 and cohens_d < 0:
        print('  ✅ 仮説は支持された')
        print('  AI GAP = 0 と予測された問題は、人間にとってもGAPが有意に小さい')
        print('  → 推薦システムとして機能している可能性がある')
    elif u_p >= 0.05:
        print('  ❌ 仮説は支持されなかった')
        print('  AI GAP = 0 と AI GAP ≠ 0 で、Human GAPに有意差がない')
    else:
        print('  ⚠️ 逆の結果')
        print('  AI GAP = 0 の方が、Human GAPが大きい傾向がある')

    # 14. AI自信度予測 vs 人間GAPの可視化
    visualize_ai_confidence_vs_human_gap(df)

def visualize_ai_confidence_vs_human_gap(df):
    """AIの自信度予測スコアと実際の人間のGAPスコアをグラフと表で可視化"""
    print('\n' + '=' * 60)
    print('📊 AI自信度予測 vs 人間GAPの可視化')
    print('=' * 60)
    
    # 出力ディレクトリ作成
    os.makedirs('output', exist_ok=True)
    
    # --- 表1: AI予測自信度別の人間GAP統計 ---
    print('\n【表1: AI予測自信度別の人間GAP統計】')
    print('-' * 70)
    
    ai_conf_analysis = df.groupby('ai_predicted_confidence').agg({
        'human_gap': ['count', 'mean', 'std', lambda x: x.abs().mean()],
        'confidence': 'mean'
    }).round(3)
    ai_conf_analysis.columns = ['件数', 'Human GAP平均', 'Human GAP標準偏差', '|Human GAP|平均', '実際の自信度平均']
    ai_conf_analysis = ai_conf_analysis.sort_index()
    print(ai_conf_analysis.to_string())
    
    # --- 表2: クロス集計表 (AI予測自信度 × Human GAP) ---
    print('\n【表2: クロス集計表 (AI予測自信度 × Human GAP)】')
    print('-' * 70)
    
    # Human GAPをカテゴリ化
    df['human_gap_cat'] = pd.cut(df['human_gap'], 
                                  bins=[-np.inf, -3, -1, 1, 3, np.inf],
                                  labels=['<-3', '-3~-1', '-1~1', '1~3', '>3'])
    
    cross_tab = pd.crosstab(df['ai_predicted_confidence'], df['human_gap_cat'], margins=True)
    print(cross_tab.to_string())
    
    # --- 表3: 相関・統計サマリー ---
    print('\n【表3: AI予測自信度と人間GAPの相関統計】')
    print('-' * 70)
    
    pearson_r, pearson_p = pearsonr(df['ai_predicted_confidence'], df['human_gap'])
    spearman_r, spearman_p = spearmanr(df['ai_predicted_confidence'], df['human_gap'])
    
    print(f'  Pearson相関係数: r = {pearson_r:.4f} (p = {pearson_p:.4g})')
    print(f'  Spearman相関係数: ρ = {spearman_r:.4f} (p = {spearman_p:.4g})')
    print(f'  決定係数 (R²): {pearson_r**2:.4f}')
    
    # --- グラフ1: 散布図 (AI予測自信度 vs Human GAP) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. 散布図 with 回帰線
    ax1 = axes[0, 0]
    ax1.scatter(df['ai_predicted_confidence'], df['human_gap'], alpha=0.5, edgecolors='k', linewidth=0.5)
    
    # 回帰線
    z = np.polyfit(df['ai_predicted_confidence'], df['human_gap'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['ai_predicted_confidence'].min(), df['ai_predicted_confidence'].max(), 100)
    ax1.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'回帰線 (y={z[0]:.3f}x+{z[1]:.3f})')
    
    ax1.set_xlabel('AI予測自信度', fontsize=12)
    ax1.set_ylabel('人間GAP (自信度 - 挑戦度)', fontsize=12)
    ax1.set_title(f'AI予測自信度 vs 人間GAP\n(r={pearson_r:.3f}, p={pearson_p:.4g})', fontsize=14)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 箱ひげ図 (AI予測自信度別のHuman GAP分布)
    ax2 = axes[0, 1]
    ai_conf_values = sorted(df['ai_predicted_confidence'].unique())
    box_data = [df[df['ai_predicted_confidence'] == val]['human_gap'].values for val in ai_conf_values]
    bp = ax2.boxplot(box_data, labels=ai_conf_values, patch_artist=True)
    
    # 色付け
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(ai_conf_values)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.set_xlabel('AI予測自信度', fontsize=12)
    ax2.set_ylabel('人間GAP', fontsize=12)
    ax2.set_title('AI予測自信度別の人間GAP分布', fontsize=14)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='GAP=0')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. ヒートマップ (AI予測自信度 × 実際の自信度)
    ax3 = axes[1, 0]
    heatmap_data = pd.crosstab(df['ai_predicted_confidence'], df['confidence'])
    im = ax3.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto')
    
    ax3.set_xticks(range(len(heatmap_data.columns)))
    ax3.set_yticks(range(len(heatmap_data.index)))
    ax3.set_xticklabels(heatmap_data.columns)
    ax3.set_yticklabels(heatmap_data.index)
    ax3.set_xlabel('実際の自信度', fontsize=12)
    ax3.set_ylabel('AI予測自信度', fontsize=12)
    ax3.set_title('AI予測自信度 × 実際の自信度 (件数)', fontsize=14)
    
    # 値を表示
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            val = heatmap_data.values[i, j]
            color = 'white' if val > heatmap_data.values.max() / 2 else 'black'
            ax3.text(j, i, str(val), ha='center', va='center', color=color, fontsize=9)
    
    plt.colorbar(im, ax=ax3, label='件数')
    
    # 4. 棒グラフ (AI予測自信度別の|Human GAP|平均)
    ax4 = axes[1, 1]
    ai_conf_stats = df.groupby('ai_predicted_confidence')['human_gap'].agg(['mean', lambda x: x.abs().mean(), 'count'])
    ai_conf_stats.columns = ['GAP平均', '|GAP|平均', '件数']
    
    x = np.arange(len(ai_conf_stats.index))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, ai_conf_stats['GAP平均'], width, label='GAP平均', color='steelblue', alpha=0.8)
    bars2 = ax4.bar(x + width/2, ai_conf_stats['|GAP|平均'], width, label='|GAP|平均', color='coral', alpha=0.8)
    
    ax4.set_xlabel('AI予測自信度', fontsize=12)
    ax4.set_ylabel('人間GAP', fontsize=12)
    ax4.set_title('AI予測自信度別の人間GAP統計', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels(ai_conf_stats.index)
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 件数をバーの上に表示
    for i, (bar, count) in enumerate(zip(bars2, ai_conf_stats['件数'])):
        ax4.annotate(f'n={count}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('output/ai_confidence_vs_human_gap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('\n✅ グラフを output/ai_confidence_vs_human_gap.png に保存しました')
    
    # --- 追加グラフ: AI GAP vs Human GAP ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. 散布図 (AI GAP vs Human GAP)
    ax1 = axes2[0]
    ax1.scatter(df['ai_gap'], df['human_gap'], alpha=0.5, edgecolors='k', linewidth=0.5)
    
    # 回帰線
    z2 = np.polyfit(df['ai_gap'], df['human_gap'], 1)
    p2 = np.poly1d(z2)
    x_line2 = np.linspace(df['ai_gap'].min(), df['ai_gap'].max(), 100)
    ax1.plot(x_line2, p2(x_line2), 'r-', linewidth=2, label=f'回帰線 (y={z2[0]:.3f}x+{z2[1]:.3f})')
    
    # 対角線 (完全一致)
    diag_min = min(df['ai_gap'].min(), df['human_gap'].min())
    diag_max = max(df['ai_gap'].max(), df['human_gap'].max())
    ax1.plot([diag_min, diag_max], [diag_min, diag_max], 'g--', linewidth=2, alpha=0.7, label='完全一致線 (y=x)')
    
    ai_gap_r, ai_gap_p = pearsonr(df['ai_gap'], df['human_gap'])
    ax1.set_xlabel('AI予測GAP', fontsize=12)
    ax1.set_ylabel('人間GAP', fontsize=12)
    ax1.set_title(f'AI予測GAP vs 人間GAP\n(r={ai_gap_r:.3f}, p={ai_gap_p:.4g})', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. ヒートマップ (AI GAP × Human GAP)
    ax2 = axes2[1]
    heatmap_gap = pd.crosstab(df['ai_gap'], df['human_gap'])
    im2 = ax2.imshow(heatmap_gap.values, cmap='YlOrRd', aspect='auto')
    
    ax2.set_xticks(range(len(heatmap_gap.columns)))
    ax2.set_yticks(range(len(heatmap_gap.index)))
    ax2.set_xticklabels(heatmap_gap.columns)
    ax2.set_yticklabels(heatmap_gap.index)
    ax2.set_xlabel('人間GAP', fontsize=12)
    ax2.set_ylabel('AI予測GAP', fontsize=12)
    ax2.set_title('AI予測GAP × 人間GAP (件数)', fontsize=14)
    
    plt.colorbar(im2, ax=ax2, label='件数')
    
    plt.tight_layout()
    plt.savefig('output/ai_gap_vs_human_gap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('✅ グラフを output/ai_gap_vs_human_gap.png に保存しました')
    
    # カテゴリ列を削除
    df.drop('human_gap_cat', axis=1, inplace=True)

def analyze_adjusted_challenge_gap(df):
    """AIの予測挑戦度を-0.51調整した仮のGAPと人間GAPの関係を分析"""
    print('\n' + '=' * 60)
    print('📊 仮説検証: AI予測挑戦度を-0.51調整した仮GAP分析')
    print('=' * 60)
    
    # 仮の挑戦度 = ai_predicted_challenge - 0.51
    df['adjusted_challenge'] = df['ai_predicted_challenge'] - 0.51
    
    # 仮のGAP = ai_predicted_confidence - adjusted_challenge
    df['adjusted_ai_gap'] = df['ai_predicted_confidence'] - df['adjusted_challenge']
    # これは ai_gap + 0.51 と同等
    
    print('\n【調整の概要】')
    print('  調整式: 仮挑戦度 = AI予測挑戦度 - 0.51')
    print('  仮GAP = AI予測自信度 - 仮挑戦度')
    print('        = AI予測自信度 - (AI予測挑戦度 - 0.51)')
    print('        = AI GAP + 0.51')
    
    print('\n【基本統計】')
    print(f'  元のAI GAP:')
    print(f'    平均: {df["ai_gap"].mean():.3f}')
    print(f'    標準偏差: {df["ai_gap"].std():.3f}')
    print(f'  調整後の仮GAP:')
    print(f'    平均: {df["adjusted_ai_gap"].mean():.3f}')
    print(f'    標準偏差: {df["adjusted_ai_gap"].std():.3f}')
    print(f'  人間GAP:')
    print(f'    平均: {df["human_gap"].mean():.3f}')
    print(f'    標準偏差: {df["human_gap"].std():.3f}')
    
    # --- 相関分析 ---
    print('\n【相関分析】')
    print('-' * 50)
    
    # 元のAI GAP vs Human GAP
    original_r, original_p = pearsonr(df['ai_gap'], df['human_gap'])
    # 調整後GAP vs Human GAP
    adjusted_r, adjusted_p = pearsonr(df['adjusted_ai_gap'], df['human_gap'])
    
    print(f'  元のAI GAP vs Human GAP:')
    print(f'    Pearson r = {original_r:.4f} (p = {original_p:.4g})')
    print(f'    R² = {original_r**2:.4f}')
    
    print(f'\n  調整後仮GAP vs Human GAP:')
    print(f'    Pearson r = {adjusted_r:.4f} (p = {adjusted_p:.4g})')
    print(f'    R² = {adjusted_r**2:.4f}')
    
    print(f'\n  → 相関係数は同じ（シフトしただけなので変わらない）')
    
    # --- 予測精度の比較 ---
    print('\n【予測精度の比較】')
    print('-' * 50)
    
    # MAE
    original_mae = np.abs(df['ai_gap'] - df['human_gap']).mean()
    adjusted_mae = np.abs(df['adjusted_ai_gap'] - df['human_gap']).mean()
    
    # 完全一致率（整数に丸めて比較）
    original_exact = (df['ai_gap'].round() == df['human_gap']).mean() * 100
    adjusted_exact = (df['adjusted_ai_gap'].round() == df['human_gap']).mean() * 100
    
    # ±1以内
    original_within1 = (np.abs(df['ai_gap'] - df['human_gap']) <= 1).mean() * 100
    adjusted_within1 = (np.abs(df['adjusted_ai_gap'] - df['human_gap']) <= 1).mean() * 100
    
    print(f'  {"指標":<20} {"元のAI GAP":<15} {"調整後仮GAP":<15} {"差分":<10}')
    print(f'  {"-"*60}')
    print(f'  {"MAE":<20} {original_mae:<15.3f} {adjusted_mae:<15.3f} {adjusted_mae - original_mae:<+10.3f}')
    print(f'  {"完全一致率 (%)":<20} {original_exact:<15.1f} {adjusted_exact:<15.1f} {adjusted_exact - original_exact:<+10.1f}')
    print(f'  {"±1以内 (%)":<20} {original_within1:<15.1f} {adjusted_within1:<15.1f} {adjusted_within1 - original_within1:<+10.1f}')
    
    # --- バイアス分析 ---
    print('\n【バイアス分析（予測 - 実測）】')
    print('-' * 50)
    
    original_bias = (df['ai_gap'] - df['human_gap']).mean()
    adjusted_bias = (df['adjusted_ai_gap'] - df['human_gap']).mean()
    
    print(f'  元のAI GAPのバイアス: {original_bias:+.3f}')
    print(f'  調整後仮GAPのバイアス: {adjusted_bias:+.3f}')
    print(f'\n  → 0.51の調整により、バイアスが {adjusted_bias - original_bias:+.3f} 変化')
    
    # --- 可視化 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. 散布図: 元のAI GAP vs Human GAP
    ax1 = axes[0, 0]
    ax1.scatter(df['ai_gap'], df['human_gap'], alpha=0.5, edgecolors='k', linewidth=0.5)
    z1 = np.polyfit(df['ai_gap'], df['human_gap'], 1)
    p1 = np.poly1d(z1)
    x_range = np.linspace(df['ai_gap'].min(), df['ai_gap'].max(), 100)
    ax1.plot(x_range, p1(x_range), 'r-', linewidth=2, label=f'回帰線 (y={z1[0]:.3f}x+{z1[1]:.3f})')
    ax1.plot([-6, 6], [-6, 6], 'g--', alpha=0.7, label='y=x')
    ax1.set_xlabel('元のAI GAP', fontsize=12)
    ax1.set_ylabel('人間GAP', fontsize=12)
    ax1.set_title(f'元のAI GAP vs 人間GAP\n(r={original_r:.3f}, MAE={original_mae:.2f})', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-6, 6)
    ax1.set_ylim(-6, 6)
    
    # 2. 散布図: 調整後GAP vs Human GAP
    ax2 = axes[0, 1]
    ax2.scatter(df['adjusted_ai_gap'], df['human_gap'], alpha=0.5, edgecolors='k', linewidth=0.5, color='orange')
    z2 = np.polyfit(df['adjusted_ai_gap'], df['human_gap'], 1)
    p2 = np.poly1d(z2)
    x_range2 = np.linspace(df['adjusted_ai_gap'].min(), df['adjusted_ai_gap'].max(), 100)
    ax2.plot(x_range2, p2(x_range2), 'r-', linewidth=2, label=f'回帰線 (y={z2[0]:.3f}x+{z2[1]:.3f})')
    ax2.plot([-6, 6], [-6, 6], 'g--', alpha=0.7, label='y=x')
    ax2.set_xlabel('調整後仮GAP (AI GAP + 0.51)', fontsize=12)
    ax2.set_ylabel('人間GAP', fontsize=12)
    ax2.set_title(f'調整後仮GAP vs 人間GAP\n(r={adjusted_r:.3f}, MAE={adjusted_mae:.2f})', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-6, 6)
    ax2.set_ylim(-6, 6)
    
    # 3. 残差分布の比較
    ax3 = axes[1, 0]
    original_residuals = df['ai_gap'] - df['human_gap']
    adjusted_residuals = df['adjusted_ai_gap'] - df['human_gap']
    
    ax3.hist(original_residuals, bins=20, alpha=0.6, label=f'元のAI GAP (mean={original_bias:.2f})', color='blue')
    ax3.hist(adjusted_residuals, bins=20, alpha=0.6, label=f'調整後仮GAP (mean={adjusted_bias:.2f})', color='orange')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='誤差=0')
    ax3.set_xlabel('予測誤差 (AI - Human)', fontsize=12)
    ax3.set_ylabel('頻度', fontsize=12)
    ax3.set_title('予測誤差の分布比較', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 調整値別のMAE変化
    ax4 = axes[1, 1]
    adjustments = np.arange(-1.0, 1.1, 0.1)
    maes = []
    for adj in adjustments:
        adjusted_gap = df['ai_gap'] + adj
        mae = np.abs(adjusted_gap - df['human_gap']).mean()
        maes.append(mae)
    
    ax4.plot(adjustments, maes, 'b-o', linewidth=2, markersize=6)
    ax4.axvline(x=0.51, color='red', linestyle='--', linewidth=2, label='提案調整値 (0.51)')
    min_adj = adjustments[np.argmin(maes)]
    ax4.axvline(x=min_adj, color='green', linestyle='--', linewidth=2, label=f'最適調整値 ({min_adj:.2f})')
    ax4.set_xlabel('調整値', fontsize=12)
    ax4.set_ylabel('MAE', fontsize=12)
    ax4.set_title('調整値とMAEの関係', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/adjusted_gap_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('\n✅ グラフを output/adjusted_gap_analysis.png に保存しました')
    
    # --- 最適な調整値の探索 ---
    print('\n【最適調整値の探索】')
    print('-' * 50)
    optimal_adjustment = adjustments[np.argmin(maes)]
    min_mae = min(maes)
    print(f'  最適な調整値: {optimal_adjustment:.2f}')
    print(f'  その時のMAE: {min_mae:.3f}')
    print(f'  提案値(0.51)でのMAE: {adjusted_mae:.3f}')
    
    # クリーンアップ
    df.drop(['adjusted_challenge', 'adjusted_ai_gap'], axis=1, inplace=True)


def analyze_precision_recall_for_adjusted_gap(df):
    """調整後GAPの適合率・再現率分析"""
    print('\n' + '=' * 60)
    print('📊 調整後GAP (AI GAP + 0.51) の適合率・再現率分析')
    print('=' * 60)
    
    # 調整後GAPを計算
    df = df.copy()
    df['adjusted_ai_gap'] = df['ai_gap'] + 0.51
    
    # 閾値リスト
    thresholds = [0, 1, 2]
    
    print('\n【分析設定】')
    print('  調整後仮GAP = AI GAP + 0.51')
    print('  「適切な問題」の定義: |GAP| ≤ 閾値')
    print('  - 適合率(Precision): 予測が「適切」の中で、実際に「適切」だった割合')
    print('  - 再現率(Recall): 実際に「適切」の中で、予測も「適切」だった割合')
    print('  - F1スコア: 適合率と再現率の調和平均')
    
    results = []
    
    for threshold in thresholds:
        print(f'\n{"="*60}')
        print(f'【閾値: |GAP| ≤ {threshold}】')
        print('='*60)
        
        # 元のAI GAP
        original_pred_positive = np.abs(df['ai_gap']) <= threshold
        # 調整後GAP
        adjusted_pred_positive = np.abs(df['adjusted_ai_gap']) <= threshold
        # 人間GAP (正解)
        actual_positive = np.abs(df['human_gap']) <= threshold
        
        # --- 元のAI GAPの分析 ---
        orig_tp = (original_pred_positive & actual_positive).sum()
        orig_fp = (original_pred_positive & ~actual_positive).sum()
        orig_fn = (~original_pred_positive & actual_positive).sum()
        orig_tn = (~original_pred_positive & ~actual_positive).sum()
        
        orig_precision = orig_tp / (orig_tp + orig_fp) if (orig_tp + orig_fp) > 0 else 0
        orig_recall = orig_tp / (orig_tp + orig_fn) if (orig_tp + orig_fn) > 0 else 0
        orig_f1 = 2 * orig_precision * orig_recall / (orig_precision + orig_recall) if (orig_precision + orig_recall) > 0 else 0
        orig_accuracy = (orig_tp + orig_tn) / len(df)
        
        # --- 調整後GAPの分析 ---
        adj_tp = (adjusted_pred_positive & actual_positive).sum()
        adj_fp = (adjusted_pred_positive & ~actual_positive).sum()
        adj_fn = (~adjusted_pred_positive & actual_positive).sum()
        adj_tn = (~adjusted_pred_positive & ~actual_positive).sum()
        
        adj_precision = adj_tp / (adj_tp + adj_fp) if (adj_tp + adj_fp) > 0 else 0
        adj_recall = adj_tp / (adj_tp + adj_fn) if (adj_tp + adj_fn) > 0 else 0
        adj_f1 = 2 * adj_precision * adj_recall / (adj_precision + adj_recall) if (adj_precision + adj_recall) > 0 else 0
        adj_accuracy = (adj_tp + adj_tn) / len(df)
        
        print(f'\n  【データ分布】')
        print(f'    人間|GAP|≤{threshold}: {actual_positive.sum()}件 ({actual_positive.mean()*100:.1f}%)')
        print(f'    元AI|GAP|≤{threshold}: {original_pred_positive.sum()}件 ({original_pred_positive.mean()*100:.1f}%)')
        print(f'    調整後|GAP|≤{threshold}: {adjusted_pred_positive.sum()}件 ({adjusted_pred_positive.mean()*100:.1f}%)')
        
        print(f'\n  【混同行列 - 元のAI GAP】')
        print(f'                        実際')
        print(f'                   適切(|GAP|≤{threshold})  不適切')
        print(f'    予測 適切         {orig_tp:>5}         {orig_fp:>5}')
        print(f'         不適切       {orig_fn:>5}         {orig_tn:>5}')
        
        print(f'\n  【混同行列 - 調整後GAP】')
        print(f'                        実際')
        print(f'                   適切(|GAP|≤{threshold})  不適切')
        print(f'    予測 適切         {adj_tp:>5}         {adj_fp:>5}')
        print(f'         不適切       {adj_fn:>5}         {adj_tn:>5}')
        
        print(f'\n  【評価指標の比較】')
        print(f'    {"指標":<15} {"元のAI GAP":<15} {"調整後GAP":<15} {"差分":<10}')
        print(f'    {"-"*55}')
        print(f'    {"適合率(Precision)":<15} {orig_precision*100:<15.1f} {adj_precision*100:<15.1f} {(adj_precision-orig_precision)*100:<+10.1f}%')
        print(f'    {"再現率(Recall)":<15} {orig_recall*100:<15.1f} {adj_recall*100:<15.1f} {(adj_recall-orig_recall)*100:<+10.1f}%')
        print(f'    {"F1スコア":<15} {orig_f1*100:<15.1f} {adj_f1*100:<15.1f} {(adj_f1-orig_f1)*100:<+10.1f}%')
        print(f'    {"正解率(Accuracy)":<15} {orig_accuracy*100:<15.1f} {adj_accuracy*100:<15.1f} {(adj_accuracy-orig_accuracy)*100:<+10.1f}%')
        
        results.append({
            'threshold': threshold,
            'orig_precision': orig_precision,
            'orig_recall': orig_recall,
            'orig_f1': orig_f1,
            'adj_precision': adj_precision,
            'adj_recall': adj_recall,
            'adj_f1': adj_f1,
        })
    
    # --- 可視化 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. 適合率・再現率の比較（閾値別）
    ax1 = axes[0, 0]
    x = np.arange(len(thresholds))
    width = 0.2
    
    orig_p = [r['orig_precision'] * 100 for r in results]
    orig_r = [r['orig_recall'] * 100 for r in results]
    adj_p = [r['adj_precision'] * 100 for r in results]
    adj_r = [r['adj_recall'] * 100 for r in results]
    
    bars1 = ax1.bar(x - 1.5*width, orig_p, width, label='元AI 適合率', color='blue', alpha=0.7)
    bars2 = ax1.bar(x - 0.5*width, orig_r, width, label='元AI 再現率', color='blue', alpha=0.4)
    bars3 = ax1.bar(x + 0.5*width, adj_p, width, label='調整後 適合率', color='orange', alpha=0.7)
    bars4 = ax1.bar(x + 1.5*width, adj_r, width, label='調整後 再現率', color='orange', alpha=0.4)
    
    ax1.set_xlabel('閾値 (|GAP| ≤)', fontsize=12)
    ax1.set_ylabel('割合 (%)', fontsize=12)
    ax1.set_title('適合率・再現率の比較（閾値別）', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'|GAP|≤{t}' for t in thresholds])
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 100)
    
    # 値をバーの上に表示
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    # 2. F1スコアの比較
    ax2 = axes[0, 1]
    orig_f1 = [r['orig_f1'] * 100 for r in results]
    adj_f1 = [r['adj_f1'] * 100 for r in results]
    
    bars1 = ax2.bar(x - width/2, orig_f1, width, label='元のAI GAP', color='blue', alpha=0.7)
    bars2 = ax2.bar(x + width/2, adj_f1, width, label='調整後GAP', color='orange', alpha=0.7)
    
    ax2.set_xlabel('閾値 (|GAP| ≤)', fontsize=12)
    ax2.set_ylabel('F1スコア (%)', fontsize=12)
    ax2.set_title('F1スコアの比較（閾値別）', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'|GAP|≤{t}' for t in thresholds])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 100)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
    
    # 3. |GAP|≤1 の詳細分析（ベン図的表現）
    ax3 = axes[1, 0]
    threshold = 1
    pred_positive = np.abs(df['adjusted_ai_gap']) <= threshold
    actual_positive_mask = np.abs(df['human_gap']) <= threshold
    
    tp = (pred_positive & actual_positive_mask).sum()
    fp = (pred_positive & ~actual_positive_mask).sum()
    fn = (~pred_positive & actual_positive_mask).sum()
    tn = (~pred_positive & ~actual_positive_mask).sum()
    
    categories = ['TP\n(正しく適切)', 'FP\n(誤って適切)', 'FN\n(見逃し)', 'TN\n(正しく不適切)']
    values = [tp, fp, fn, tn]
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
    
    bars = ax3.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('件数', fontsize=12)
    ax3.set_title(f'調整後GAP |GAP|≤1 の分類結果\n(n={len(df)})', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax3.annotate(f'{val}\n({val/len(df)*100:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 4. 調整値別の F1スコア変化（|GAP|≤1の場合）
    ax4 = axes[1, 1]
    adjustments = np.arange(-1.0, 1.1, 0.1)
    f1_scores = []
    precisions = []
    recalls = []
    
    threshold = 1
    actual_positive_mask = np.abs(df['human_gap']) <= threshold
    
    for adj in adjustments:
        adjusted_gap = df['ai_gap'] + adj
        pred_positive = np.abs(adjusted_gap) <= threshold
        
        tp = (pred_positive & actual_positive_mask).sum()
        fp = (pred_positive & ~actual_positive_mask).sum()
        fn = (~pred_positive & actual_positive_mask).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision * 100)
        recalls.append(recall * 100)
        f1_scores.append(f1 * 100)
    
    ax4.plot(adjustments, precisions, 'b-', linewidth=2, label='適合率', marker='o', markersize=4)
    ax4.plot(adjustments, recalls, 'g-', linewidth=2, label='再現率', marker='s', markersize=4)
    ax4.plot(adjustments, f1_scores, 'r-', linewidth=2.5, label='F1スコア', marker='^', markersize=4)
    
    ax4.axvline(x=0.51, color='orange', linestyle='--', linewidth=2, label='提案調整値 (0.51)')
    ax4.axvline(x=0, color='gray', linestyle=':', linewidth=1.5, label='調整なし (0)')
    
    best_adj = adjustments[np.argmax(f1_scores)]
    ax4.axvline(x=best_adj, color='purple', linestyle='--', linewidth=2, label=f'最適調整値 ({best_adj:.2f})')
    
    ax4.set_xlabel('調整値', fontsize=12)
    ax4.set_ylabel('スコア (%)', fontsize=12)
    ax4.set_title('調整値と適合率・再現率・F1スコアの関係\n(|GAP|≤1)', fontsize=14)
    ax4.legend(loc='lower right')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('output/precision_recall_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('\n✅ グラフを output/precision_recall_analysis.png に保存しました')
    
    # --- サマリー ---
    print('\n' + '=' * 60)
    print('📋 サマリー')
    print('=' * 60)
    
    print('\n【|GAP|≤1 での推薦システム性能】')
    t1_result = results[1]  # threshold=1
    print(f'  元のAI GAP:')
    print(f'    適合率: {t1_result["orig_precision"]*100:.1f}%')
    print(f'    再現率: {t1_result["orig_recall"]*100:.1f}%')
    print(f'    F1スコア: {t1_result["orig_f1"]*100:.1f}%')
    print(f'  調整後GAP (AI GAP + 0.51):')
    print(f'    適合率: {t1_result["adj_precision"]*100:.1f}%')
    print(f'    再現率: {t1_result["adj_recall"]*100:.1f}%')
    print(f'    F1スコア: {t1_result["adj_f1"]*100:.1f}%')
    
    print(f'\n【最適調整値の探索結果 (|GAP|≤1)】')
    print(f'  F1スコア最大化の調整値: {best_adj:.2f}')
    print(f'  その時のF1スコア: {max(f1_scores):.1f}%')
    
    print('\n【解釈】')
    if t1_result["adj_f1"] > t1_result["orig_f1"]:
        print('  ✅ 0.51の調整によりF1スコアが改善')
    else:
        print('  ❌ 0.51の調整ではF1スコアは改善しない')
    
    if t1_result["adj_precision"] > t1_result["orig_precision"]:
        print('  ✅ 適合率が改善（誤推薦が減少）')
    else:
        print('  ❌ 適合率は低下')
    
    if t1_result["adj_recall"] > t1_result["orig_recall"]:
        print('  ✅ 再現率が改善（見逃しが減少）')
    else:
        print('  ❌ 再現率は低下')


if __name__ == "__main__":
    main()
    
    # 追加分析の実行
    sessions_df = pd.read_csv('data/sessions.csv')
    df = pd.read_csv('data/responses.csv')
    analyze_adjusted_challenge_gap(df)
    
    # 適合率・再現率分析
    df = pd.read_csv('data/responses.csv')  # 再読み込み
    analyze_precision_recall_for_adjusted_gap(df)
