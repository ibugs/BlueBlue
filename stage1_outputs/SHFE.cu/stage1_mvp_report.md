# Stage 1 MVP Report

## Data Scope
- input_dir: `/Users/wangrendong/Projects/BlueBlue/orderflow_data/SHFE.cu`
- start_date: `2023-05-27`
- contracts: `34`
- rows: `72614`
- datetime_min: `2023-05-27 00:00:00`
- datetime_max: `2026-04-14 23:55:00`

## Feature List
`bar_return`, `range_ticks`, `body_ratio`, `close_position`, `upper_shadow_ratio`, `lower_shadow_ratio`, `delta_strength`, `delta_zscore_20`, `volume_zscore_20`, `cvd_change_10`, `poc_distance_ticks`, `poc_shift_ticks`, `open_interest_change`, `open_interest_zscore_20`, `trend_return_12`, `volatility_20`, `price_delta_agreement`

## Label Definition
- primary label: `future_return_5`
- strategy label: next-bar open entry, 5-bar close exit, 2-tick roundtrip cost.

## Top Features By Train Spearman IC
| feature              |   non_null_count |   valid_count |   coverage |   pearson_ic_5 |   spearman_ic_5 |   full_pearson_ic_5 |   full_spearman_ic_5 |   test_spearman_ic_5 |   ic_direction |   quintile_spread_mean_return |   top_quintile_win_rate |   top_quintile_mean_mfe |   top_quintile_mean_mae |
|:---------------------|-----------------:|--------------:|-----------:|---------------:|----------------:|--------------------:|---------------------:|---------------------:|---------------:|------------------------------:|------------------------:|------------------------:|------------------------:|
| close_position       |            72614 |         72444 |   0.997659 |    -0.0160301  |      -0.0263738 |        -0.0108755   |           -0.0249903 |          -0.0221478  |             -1 |                  -7.82936e-05 |                0.457357 |              0.0020153  |             -0.00201152 |
| volatility_20        |            71934 |         71764 |   0.988294 |    -0.00352237 |      -0.0234585 |        -0.000411645 |           -0.0127126 |           0.00162098 |             -1 |                  -0.000125563 |                0.479342 |              0.00306284 |             -0.00327002 |
| poc_distance_ticks   |            72614 |         72444 |   0.997659 |    -0.0183023  |      -0.0210857 |        -0.00283667  |           -0.0150703 |          -0.00439957 |             -1 |                 nan           |              nan        |            nan          |            nan          |
| bar_return           |            72614 |         72444 |   0.997659 |    -0.0107051  |      -0.017237  |         0.00473892  |           -0.0207021 |          -0.027657   |             -1 |                  -1.61458e-05 |                0.465664 |              0.00237907 |             -0.00235947 |
| lower_shadow_ratio   |            72614 |         72444 |   0.997659 |    -0.0146117  |      -0.0143887 |        -0.010798    |           -0.0112322 |          -0.00471484 |             -1 |                 nan           |              nan        |            nan          |            nan          |
| cvd_change_10        |            72274 |         72104 |   0.992977 |    -0.00266227 |      -0.0128707 |         0.00816345  |           -0.0155224 |          -0.0202657  |             -1 |                   3.93071e-05 |                0.470858 |              0.00226064 |             -0.00219154 |
| upper_shadow_ratio   |            72614 |         72444 |   0.997659 |     0.0113118  |       0.0121411 |         0.00378867  |            0.0064828 |          -0.0056276  |              1 |                 nan           |              nan        |            nan          |            nan          |
| open_interest_change |            72580 |         72410 |   0.997191 |     0.011804   |       0.0118577 |         0.0317894   |            0.0073734 |           0.00145899 |              1 |                   9.12637e-05 |                0.483567 |              0.00233432 |             -0.00215459 |

## Most Negative Features By Train Spearman IC
| feature            |   non_null_count |   valid_count |   coverage |   pearson_ic_5 |   spearman_ic_5 |   full_pearson_ic_5 |   full_spearman_ic_5 |   test_spearman_ic_5 |   ic_direction |   quintile_spread_mean_return |   top_quintile_win_rate |   top_quintile_mean_mfe |   top_quintile_mean_mae |
|:-------------------|-----------------:|--------------:|-----------:|---------------:|----------------:|--------------------:|---------------------:|---------------------:|---------------:|------------------------------:|------------------------:|------------------------:|------------------------:|
| close_position     |            72614 |         72444 |   0.997659 |   -0.0160301   |      -0.0263738 |        -0.0108755   |          -0.0249903  |         -0.0221478   |             -1 |                  -7.82936e-05 |                0.457357 |              0.0020153  |             -0.00201152 |
| volatility_20      |            71934 |         71764 |   0.988294 |   -0.00352237  |      -0.0234585 |        -0.000411645 |          -0.0127126  |          0.00162098  |             -1 |                  -0.000125563 |                0.479342 |              0.00306284 |             -0.00327002 |
| poc_distance_ticks |            72614 |         72444 |   0.997659 |   -0.0183023   |      -0.0210857 |        -0.00283667  |          -0.0150703  |         -0.00439957  |             -1 |                 nan           |              nan        |            nan          |            nan          |
| bar_return         |            72614 |         72444 |   0.997659 |   -0.0107051   |      -0.017237  |         0.00473892  |          -0.0207021  |         -0.027657    |             -1 |                  -1.61458e-05 |                0.465664 |              0.00237907 |             -0.00235947 |
| lower_shadow_ratio |            72614 |         72444 |   0.997659 |   -0.0146117   |      -0.0143887 |        -0.010798    |          -0.0112322  |         -0.00471484  |             -1 |                 nan           |              nan        |            nan          |            nan          |
| cvd_change_10      |            72274 |         72104 |   0.992977 |   -0.00266227  |      -0.0128707 |         0.00816345  |          -0.0155224  |         -0.0202657   |             -1 |                   3.93071e-05 |                0.470858 |              0.00226064 |             -0.00219154 |
| range_ticks        |            72614 |         72444 |   0.997659 |   -0.0220761   |      -0.0117113 |        -0.0331265   |          -0.00892845 |         -0.00892699  |             -1 |                  -8.73349e-05 |                0.487073 |              0.00324985 |             -0.00342121 |
| delta_strength     |            72614 |         72444 |   0.997659 |   -0.000316004 |      -0.011077  |         0.00265604  |          -0.00787362 |         -0.000752738 |             -1 |                  -7.94489e-06 |                0.468079 |              0.00180978 |             -0.00163096 |

## Selected MVP Strategy Features
| feature            |   spearman_ic_5 |   coverage |
|:-------------------|----------------:|-----------:|
| close_position     |      -0.0263738 |   0.997659 |
| volatility_20      |      -0.0234585 |   0.988294 |
| poc_distance_ticks |      -0.0210857 |   0.997659 |
| bar_return         |      -0.017237  |   0.997659 |
| lower_shadow_ratio |      -0.0143887 |   0.997659 |

## MVP Strategy Summary
| split   |   trades |   long_trades |   short_trades |   win_rate |   avg_net_return |   median_net_return |   total_net_return_sum |   per_trade_sharpe |   max_drawdown_sum |
|:--------|---------:|--------------:|---------------:|-----------:|-----------------:|--------------------:|-----------------------:|-------------------:|-------------------:|
| train   |     7564 |          3912 |           3652 |   0.441565 |     -0.000212211 |        -0.000153006 |              -1.60517  |           -8.72866 |          -1.61228  |
| test    |     3518 |          1581 |           1937 |   0.436043 |     -0.000259419 |        -0.000231951 |              -0.912637 |           -5.16216 |          -0.915281 |
| all     |    11082 |          5493 |           5589 |   0.439812 |     -0.000227198 |        -0.000231254 |              -2.5178   |           -9.8702  |          -2.52397  |

## First-Stage Findings
- The MVP now produces a repeatable feature-ranking and strategy-feedback loop.
- Feature signs are chosen only from the training set to avoid test leakage.
- Strategy performance should be treated as event-study feedback, not a production trading result.
- Test trades: `3518`; test avg net return: `-0.00025941918558537746`.

## Stage 2 Fix List
- Replace the simple 2-tick cost proxy with exchange fee, slippage, and realistic execution assumptions.
- Add market-regime segmentation: trend/range, high/low volatility, high/low volume.
- Add walk-forward feature selection instead of one fixed train/test split.
- Add richer order-flow features after the current 17-feature MVP is reviewed.
- Add charts for IC stability, quintile monotonicity, and strategy equity.
