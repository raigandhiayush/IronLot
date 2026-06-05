# 🏎️ Automated Live Car Auction Trading Agent

An end-to-end machine learning pipeline and context-aware dynamic bidding agent designed to predict vehicle market values and execute optimized automated trading strategies in a simulated live-auction environment.

The architecture maximizes Return on Investment (ROI) by leveraging a highly tuned **CatBoost Regressor** for asset valuation and an adaptive **Volume-Driven Capital Liquidation Strategy** to manage an initial bankroll of $500,000 against an unknown timeline.

---

## 📌 Project Architecture & Pipeline
[ Raw Auction Stream ]
│
▼
[ Feature Engineering ]  ──► (Vehicle Age, Usage Intensity, Premium Scores, etc.)
│
▼
[ CatBoost Model ]     ──► (Log-scale valuation inference)
│
▼
[ Live Auction Agent ]    ──► (Evaluates Deal Quality, Rival Aggression, & Wallet Tiers)
│
▼
[ Final Bid ]        ──► (Bids optimally or exits gracefully)

---

## 🛠️ Feature Engineering Pipeline

Tree-based models can struggle to naturally infer non-linear physical degradation or geographic demand relationships. This pipeline constructs explicit interaction terms to significantly lower your Root Mean Squared Error (RMSE):

*   **Vehicle Age (`car_age`)**: Calculates the linear deprecation horizon of the vehicle relative to the calendar model timeline (indexed to the baseline year 2026).
*   **Usage Intensity (`usage_intensity`)**: Ratios total vehicle mileage against absolute age to isolate low-mileage highway cruisers from heavily overused commuter cars.
*   **Premium Brand Score (`premium_score`)**: Maps vehicle make frequencies into a specialized multi-tier premium dictionary to establish tier pricing baselines.
*   **Regional Target Price Encoding (`state_avg_price` & `brand_avg_price`)**: Captures environmental factors (such as undercarriage rust risk within the Northern Snow/Rust Belt) and historical local auction demand variables.
*   **High-Cardinality Target Extractions**: Normalizes highly distributed categoricals (`model`, `trim`, `body`) via a clean frequency mapper serialization object (`mappings_AyushRaigandhi.pkl`).

---

## 🧬 Hyperparameter Tuning (Optuna Engine)

Hyperparameters are searched using **Optuna's Tree-structured Parzen Estimator (TPE)**, targeting the optimization of validation dataset RMSE. 

### Exploration Distribution Space:
*   `iterations`: `[500, 1000, 1500]`
*   `learning_rate`: `0.01` to `0.1` (Step: `0.01`)
*   `depth`: `4` to `8` (Integer Step)
*   `l2_leaf_reg`: `1` to `9` (Step: `2`)

*Note: The definitive 80/20 train-validation split is strictly maintained and locked prior to executing optimization trials to eliminate any internal cross-validation data leak traps.*

---

## 💰 Volume-Driven Bidding Framework

The bidding engine (`LiveAuctionAgent`) evaluates **three concurrent real-time game signals** every round to calculate its final geometric aggression increment:

1.  **Deal Quality Matrix**: $\frac{\text{Predicted Value} - \text{Current Highest Bid}}{\text{Predicted Value}}$
    Measures the instant financial upside left on the table. When high, it pushes for faster raises; when narrow, it tapers down.
2.  **Rival Aggression Metric**: $\frac{\text{Rival Increment Size}}{\text{Predicted Value}}$
    Monitors the step adjustments made by competing bots. If rivals throw massive jumps, the agent scales back to protect its margin floor. If the rival shows weakness via tiny ticks, the agent strikes with an assertive counter.
3.  **Bid-to-Value Positioning**: Monitors where the auction currently sits relative to the true asset value to smoothly transition from aggressive early-stage entries into highly conservative adjustments near the exit ceiling.

### Dynamic Capital Liquidation
Because the total number of cars to be sold across the tournament is hidden, the engine protects against under-buying by dynamically scaling its bidding rules based on its liquid wallet metrics:

| Capital Tier (Bankroll) | Liquidation Multiplier | Single-Car Allocation Cap | Strategic Mode |
| :--- | :--- | :--- | :--- |
| **High Capital** ($> \$350\text{k}$) | `1.03` | **25%** of Bankroll | **Aggressive Liquidation**: Shaves margins slightly to secure rapid asset volume before the auction stream cuts out. |
| **Mid Capital** ($\$150\text{k}$ - $\$350\text{k}$) | `1.00` | **15%** of Bankroll | **Balanced Operation**: Standard configuration optimizing baseline profit spreads. |
| **Low Capital** ($< \$150\text{k}$) | `0.93` | **10%** of Bankroll | **Hyper-Conservative**: Locks down capital. Stands back from bidding wars and demands massive profit margins. |

---

## 📂 Repository Structure

*   `agent_AyushRaigandhi.py`: Core production agent file containing feature transforms, prediction hooks, and the dynamic liquidation bidding loop.
*   `test_arena.py`: Standalone local validation sandbox simulating historical auction logs, multiple competitor profiles (Aggressive, Balanced, Cautious), and real-time wallet tracking.
*   `model_AyushRaigandhi.pkl`: Serialized final optimized CatBoost Regressor binary.
*   `mappings_AyushRaigandhi.pkl`: Dictionary tracking global target means, frequency buckets, and the expected input column sequence configuration.

---

## 🚀 Execution & Simulation Run

To validate your agent's bidding efficiency, ROI spreads, and capital liquidation behavior locally against randomized rival strategies, launch the local testing arena:

```bash
python test_arena.py
