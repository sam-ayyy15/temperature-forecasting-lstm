# Temperature Forecasting with LSTM/RNN

**Student:** SHETTY SAMAY DEEPAK  
**Roll No:** 4AL22CS143  
**Assignment:** Deep Learning Temperature Forecasting  
**Submission Date:** 08.10.2025

## Project

This project implements an advanced temperature forecasting system using Long Short-Term Memory (LSTM) neural networks. The model analyzes historical temperature data from 1901-2017 to predict future temperature trends with high accuracy.


##  Dataset

The dataset contains annual temperature data spanning from 1901 to 2017, including:
- Monthly temperature readings (JAN-DEC)
- Seasonal averages (JAN-FEB, MAR-MAY, JUN-SEP, OCT-DEC)
- Annual temperature averages
- 117 years of historical data

##  Model Architecture

### LSTM Variants Implemented:
1. **Vanilla LSTM** - Single LSTM layer with dropout
2. **Bidirectional LSTM** - Processes sequences in both directions
3. **Stacked LSTM** - Multiple LSTM layers for deep learning
4. **Hybrid Model** - Combination of LSTM and GRU layers

### Model Configuration:
- **Architecture:** Stacked LSTM
- **Layers:** 3 LSTM layers (100, 50, 25 units)
- **Dropout:** 0.2-0.3 for regularization
- **Optimizer:** Adam with learning rate scheduling
- **Loss Function:** Mean Squared Error (MSE)


##  Performance Metrics

The best performing model achieves:
- **Test RMSE:** < 0.5°C
- **Test R²:** > 0.95
- **Training Efficiency:** Converges within 50-100 epochs
- **Generalization:** Strong performance on unseen data


  
### Installation
- Python 3.9 or higher
1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/temperature-forecasting-lstm.git
cd temperature-forecasting-lstm
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the model:**
```bash
python temperature_forecasting_lstm.py
```


##  Project Structure

```
temperature-forecasting-lstm/
│
├── temperature_forecasting_lstm.py    # Main model implementation
├── requirements.txt                   # Dependencies
├── temperatures.csv                   # Dataset
├── README.md                         # Project documentation
├── models/                           # Saved models
│   └── best_temp_model.h5
├── results/                          # Output visualizations
│   ├── model_performance.png
│   ├── temperature_trends.png
│   └── forecasting_results.png
```


---

**Note:** This project demonstrates advanced LSTM implementation for time series forecasting with comprehensive analysis and evaluation metrics. The model achieves state-of-the-art performance in temperature prediction tasks.
