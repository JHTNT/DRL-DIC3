# Multi-Armed Bandit Strategy Comparison

互動式多臂賭徒問題策略比較工具，支援 6 種經典演算法。

## 📁 Project Structure

```
.
├── strategies.py                      # 核心：所有 6 種策略的共用實裝
├── streamlit_app.py                   # ⭐ 主應用：互動式 Streamlit 儀表板
├── mab_algorithms_comparison.py       # 命令列版本：對比分析 + 圖表輸出
├── mab_ab_testing_analysis.py         # 詳細分析：A/B 測試 + 對比
├── mab_regret_theory.py               # 教學模組：數學基礎 & Regret 理論
├── requirements.txt                    # Python 依賴
└── README.md                          # 本檔案
```

## 🚀 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 執行 Streamlit 應用（推薦）

```bash
streamlit run streamlit_app.py
```

這會在瀏覽器開啟互動式儀表板，你可以：

- **左側邊欄**：調整所有參數
  - 調節各臂的期望回報 (μ_A, μ_B, μ_C)
  - 設定總次數與蒙地卡羅執行次數
  - 調整各策略的超參數（ε, 溫度, UCB-c 等）
- **主頁面**：即時看到結果與圖表
  - 策略績效比較表
  - 平均回報柱狀圖
  - Regret 對比圖
  - 學習曲線（每步平均回報變化）
  - 各臂拉取分配圖
  - 排名詳細分析

### 3. 執行無 UI 版本（命令列）

如果只想看文字輸出與靜態圖表，執行：

```bash
python mab_algorithms_comparison.py
```

輸出會包括：

- 策略績效表格
- 生成 4 張 PNG 圖表到 `./charts/` 目錄：
  - `strategy_mean_reward.png` - 平均回報
  - `strategy_regret.png` - Regret 對比
  - `learning_curves.png` - 學習曲線
  - `arm_allocation.png` - 臂拉取分配

### 4. 分析模組

執行原始分析（包含詳細計算過程）：

```bash
python mab_ab_testing_analysis.py      # A/B 測試分析
python mab_regret_theory.py             # 數學推導
```

## 📊 支援的策略

| 策略 | 說明 | 超參數 |
|------|------|--------|
| **A/B Testing** | 固定時間測試 A、B，然後利用最佳臂 | A/B 測試佔比 |
| **Optimistic Initial Values** | Q-value 初始化為樂觀值，自動激勵探索 | 初始值 |
| **ε-Greedy** | 以 ε 機率隨機探索，(1-ε) 機率貪心利用 | ε (探索率) |
| **Softmax (Boltzmann)** | 溫度函數決定探索機率分佈 | 溫度 (Temperature) |
| **UCB (Upper Confidence Bound)** | 樂觀主義：選擇置信區間上界最高的臂 | c (置信係數) |
| **Thompson Sampling** | 貝氏推論：從後驗分佈採樣 | 無 (自動調整) |

## 🎛️ Streamlit 參數說明

### Bandit Parameters（左側邊欄）

- **μ_A, μ_B, μ_C**: 各臂的伯努利成功機率
  - 範圍: [0.0, 1.0]
  - 真實場景：A=0.8 (最優)、B=0.7、C=0.5

### Simulation Settings

- **Total Pulls**: 總預算或總嘗試次數
  - 範圍: [1000, 50000]
  - 預設: 10000
- **Number of Runs**: 蒙地卡羅模擬執行次數
  - 更多執行 → 更穩定的結果，但執行時間更長
  - 範圍: [10, 500]
  - 推薦: 100-200

### Strategy Hyperparameters（各策略專用參數）

**A/B Testing:**

- **A/B Test Fraction**: 花在測試階段的時間比例
  - 預設: 0.2 (20% 測試期, 80% 利用期)

**ε-Greedy:**

- **ε**: 每步探索機率
  - 小 ε (0.01-0.05): 快速收斂，但可能卡在局部最優
  - 大 ε (0.2-0.3): 強烈探索，避免卡住，但效率降低

**Softmax (Boltzmann):**

- **Temperature**: 控制探索強度
  - 高溫 (0.5-1.0): 更均勻的探索
  - 低溫 (0.01-0.1): 快速集中到好臂

**UCB:**

- **c**: 置信區間寬度係數
  - 小 c (0.5-1.0): 傾向利用
  - 大 c (2.0-5.0): 傾向探索

**Optimistic Initial Values:**

- **Initial Value**: Q(a) 初始值
  - 高初值 (0.8-2.0): 激勵探索
  - 低初值 (0.1-0.5): 稍微利用

**Thompson Sampling:**

- 無超參數，完全自適應

## 📈 輸出指標

### 表格欄位

| 欄位 | 說明 |
|------|------|
| Strategy | 策略名稱 |
| Mean Reward | 平均總回報（跨 N_runs 執行） |
| Std Dev | 標準差 |
| Regret | 與最優回報的差距 |
| A pulls | 平均拉取臂 A 次數 |
| B pulls | 平均拉取臂 B 次數 |
| C pulls | 平均拉取臂 C 次數 |

### 圖表

1. **Mean Reward + Optimal**: 柱狀圖，紅虛線標示最優值
2. **Regret**: 各策略的 Regret 大小
3. **Learning Curves**: 折線圖顯示各策略在每一步的平均回報進展
4. **Arm Allocation**: 堆疊柱狀圖，每個策略如何分配 N 次嘗試

## 💡 使用建議

### 場景 1：比較策略整體績效

1. 保持預設參數
2. 增加 `Number of Runs` 到 300-500 以提高統計穩定性
3. 按 **Run Simulation** 觀察排名

### 場景 2：調整超參數來改善特定策略

1. 選擇你感興趣的策略對應的超參數
2. 逐次微調，每次修改後運行模擬
3. 觀察 Learning Curves 看學習過程

### 場景 3：研究探索-利用權衡

1. 改變 `ε-Greedy ε` 或 `Softmax Temperature`
2. 觀察 `Arm Allocation` 圖：更高的參數 → 更多探索
3. 對比 `Learning Curves`：找到平衡點

### 場景 4：測試難度變化對策略的影響

1. 改變 μ_A, μ_B, μ_C 的間隔
   - **簡單**: μ_A=0.9, μ_B=0.5, μ_C=0.1 (大間隔)
   - **困難**: μ_A=0.75, μ_B=0.73, μ_C=0.70 (小間隔)
2. 觀察不同策略的相對表現變化

## 🔬 技術細節

### 獎勵模型

所有獎勵採用 **Bernoulli 模型**：

- 每次拉取臂 a，以機率 μ_a 得到 +1 獎勵，否則 +0

### 評估方法

- **蒙地卡羅估計**: 執行 N_runs 次獨立模擬，取平均結果
- **平均回報曲線**: 累積回報 / 當前步數，i.e., 每一步的平均回報
- **Regret**: 最優策略期望回報 - 當前策略期望回報

### 優化點

- Streamlit 使用快取減少重計算
- 策略類統一介面，易於擴展新策略
- 圖表使用 matplotlib 且透過 Streamlit 即時顯示（無需存檔）

## 🏗️ 模組架構說明

### **strategies.py** - 核心策略模塊 ⭐

代碼清理後的新核心模塊，集中所有 6 種策略的實裝，**消除重複代碼，提升可維護性**。

**設計模式**（繼承層級）：

```
BaseBanditStrategy (抽象基類)
├── ABTestingStrategy
├── OptimisticInitialValuesStrategy
├── EpsilonGreedyStrategy
├── SoftmaxBoltzmannStrategy
├── UCB1Strategy
└── ThompsonSamplingStrategy
```

**快速使用**：

```python
from strategies import EpsilonGreedyStrategy

strategy = EpsilonGreedyStrategy(
  means={'A': 0.8, 'B': 0.7, 'C': 0.5},
  total_pulls=10000,
  epsilon=0.1
)

for t in range(10000):
  arm = strategy.select_arm(t)
  reward = strategy.sample_reward(arm)
  strategy.update(arm, reward)

# 結果
assert len(strategy.average_reward_curve) == 10000
print(f"平均回報: {strategy.total_reward / 10000:.3f}")
print(f"各臂拉取次數: {strategy.counts}")
```

**為何採用這個設計？**

| 優點 | 說明 |
|------|------|
| ✅ **代碼複用** | 避免在 streamlit_app.py 和 mab_algorithms_comparison.py 中重複定義策略 |
| ✅ **易於擴展** | 加新策略只需在 strategies.py 中新增一個類別 |
| ✅ **單一真理來源** | 策略邏輯只有一份副本，修改方便 |
| ✅ **責任分離** | 策略邏輯 ← 與 → UI/命令列邏輯完全分開 |

**版本控制歷程**：

- 初版：策略定義分散在 streamlit_app.py、mab_algorithms_comparison.py 裡 → 代碼重複 ❌
- 重構版：所有策略統一至 strategies.py → 代碼乾淨，易維護 ✅
