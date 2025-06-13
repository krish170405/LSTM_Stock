# LSTM Stock Price Predictor

A PyTorch-based LSTM model for forecasting Microsoft (MSFT) closing prices using engineered features and a sliding window of past observations.


## Methodology

1. **Data Acquisition**\
   Download daily OHLCV data for MSFT from 2000-01-01 to 2010-01-01 using the `yfinance` library.

2. **Feature Engineering**

   - **Close Price**
   - **Log-Return**:
     ```math
       r_t = \ln\bigl(\tfrac{C_t}{C_{t-1}}\bigr)
     ```
   - **Momentum** (5-day percent change)
   - **SMA Deviation**: difference between close and its 5-day simple moving average
   - **Volatility**: rolling standard deviation of log-returns (5-day)
   - **Volume Ratio**: ratio of daily volume to its 5-day moving average

3. **Windowing**\
   Create overlapping windows of length \(k=5\). Each sample consists of a tensor of shape \((k, 6)\) and targets the closing price on the next day.

4. **Train/Validation/Test Split**

   - 70% training
   - 15% validation
   - 15% test

5. **Scaling**\
   Standardize all inputs and targets to zero mean and unit variance.

6. **Model Architecture**\
   A single-layer LSTM followed by a fully connected layer:

   ```python
   nn.LSTM(input_size=6, hidden_size=64, batch_first=True)
   nn.Linear(64, 1)
   ```

7. **Training**

   - Loss function: Mean Squared Error
   - Optimizer: Adam (learning rate = 1e-3)
   - Epochs: 50
   - Batch size: 32

8. **Evaluation**

   - Compute MSE, RMSE, and R² for each data split
   - Plot training and validation loss curves
   - Plot true vs. predicted closing prices over time
   - Scatter plot of predictions vs. ground truth

---

## Results

- **Training** achieves very low error (RMSE ≈ \$0.38, R² ≈ 0.985).
- **Validation** remains strong (RMSE ≈ \$0.41, R² ≈ 0.940).
- **Test** generalizes well (RMSE ≈ \$0.56, R² ≈ 0.964).

Visual summaries:

- **Loss Curves**: shows rapid convergence and stability.
- **Price Trajectories**: true and predicted lines nearly overlap.
- **Prediction Scatter**: tight clustering around the 45° line.

---

## Next Steps

- **Additional Features**: incorporate technical indicators (e.g., RSI, Bollinger Bands).
- **Hyperparameter Optimization**: perform grid or random search over window size, hidden units, dropout, and learning rate.
- **Ensembles**: combine LSTM predictions with tree-based regressors.
- **Return-Based Target**: predict log-returns directly for trading strategies.

---

