# GAN-Based Stock Price Forecasting

### Comparative Next-Day Forecasting with LSTM, GAN and Conditional GAN Variants

A research implementation for **next-day stock-price forecasting** using recurrent neural networks and adversarial learning.

The project compares a supervised **LSTM baseline** against several GAN-based forecasting architectures, including conditional models that incorporate calendar and volatility-regime information.

The current implementation focuses on **Microsoft (MSFT)** market data from **2000–2020** and evaluates forecasts using chronologically separated train, validation and test sets.

---

## Project Overview

The repository began as an implementation of a GAN-based stock forecasting approach and was subsequently extended into a broader comparative modelling pipeline.

The current system evaluates:

| Model       | Description                                                |
| ----------- | ---------------------------------------------------------- |
| LSTM        | Supervised recurrent baseline                              |
| GAN         | LSTM generator + MLP discriminator                         |
| cGAN (DOW)  | Conditional GAN using day-of-week information              |
| Stable cGAN | DOW-conditioned GAN with additional training stabilization |
| Regime cGAN | Conditional GAN using volatility-regime information        |
| Ensemble    | LSTM + cGAN prediction averaging                           |
| Ensemble++  | Multi-model prediction averaging                           |

Rather than assuming that adversarial modelling must outperform a simpler forecasting model, the project compares all approaches using the same chronological evaluation framework.

---

## Research Objective

Financial forecasting is challenging because market observations are:

* sequential;
* non-stationary;
* noisy;
* affected by changing volatility;
* influenced by both temporal structure and market regime.

A standard LSTM can directly optimize prediction error, while a GAN introduces an additional adversarial objective intended to encourage generated outputs that better resemble the observed data distribution.

This project explores whether adding adversarial learning and contextual conditioning improves next-day forecasting compared with a conventional recurrent baseline.

---

## Dataset

Market data are downloaded programmatically using **yfinance**.

The current dataset configuration is:

| Property          | Value                          |
| ----------------- | ------------------------------ |
| Asset             | Microsoft Corporation (`MSFT`) |
| Start             | 2000-01-01                     |
| End               | 2020-12-31                     |
| Forecast horizon  | 1 trading day                  |
| Historical window | 20 trading days                |
| Train split       | 90%                            |
| Validation split  | 5%                             |
| Test split        | 5%                             |

The pipeline uses chronological splitting rather than random shuffling across time.

---

## Input Features

The modelling pipeline uses seven market-derived features:

```text
high
low
open
close
volume
turnover
ma5
```

`turnover` is implemented as trading volume relative to its rolling 20-day mean.

`ma5` represents a five-day moving average of the closing price.

---

## Preprocessing Pipeline

```text
Yahoo Finance
      |
      v
OHLCV market data
      |
      +--> 5-day moving average
      |
      +--> relative volume / turnover
      |
      v
Moving-average normalization
      |
      v
Chronological train / validation / test split
      |
      v
20-day historical windows
      |
      v
Next-day close prediction
```

### Moving-Average Normalization

For each feature, the implementation constructs a rolling mean and standard deviation and converts observations into a local z-score representation.

For a feature \(x_t\):

$$
z_t =
\frac{x_t-\mu_t}{\sigma_t}
$$

where \(\mu_t\) and \(\sigma_t\) are calculated using the configured rolling window.

The current configuration uses:

```text
MAR window = 5
```

Predictions are transformed back into USD for evaluation.

---

## LSTM Baseline

The baseline model is a many-to-one LSTM regressor.

```text
20-day feature sequence
        |
        v
2-layer LSTM
hidden size = 128
        |
        v
Fully connected head
        |
        v
Next-day normalized close
```

The model is trained directly using mean-squared prediction error.

This provides an important benchmark for determining whether adversarial training contributes useful predictive information.

---

## Classical GAN

The GAN uses an **LSTM generator** and an **MLP discriminator**.

### Generator

The generator receives:

$$
X \in \mathbb{R}^{T \times F}
$$

where \(T\) is the historical window and \(F\) is the number of input features.

An LSTM encodes the historical sequence and generates the next-step feature representation.

### Discriminator

The discriminator receives the flattened historical window together with either the real or generated next-step vector.

Its task is to distinguish:

```text
real next-step observation
            vs
generated next-step observation
```

### Generator Objective

Training combines two objectives:

$$
L_G =
\lambda_{GAN}L_{adv}
+
\lambda_{MSE}L_{pred}
$$

where:

* \(L_{adv}\) is the adversarial binary-cross-entropy objective;
* \(L_{pred}\) is supervised MSE on the closing-price component.

The generator therefore optimizes both predictive accuracy and adversarial realism.

---

## Conditional GAN

The conditional GAN extends the generator and discriminator with an additional contextual vector.

The generator uses:

```text
historical market sequence
          +
condition vector
          |
          v
next-day prediction
```

The discriminator also receives the same condition when evaluating real and generated samples.

Two primary conditioning strategies are explored.

---

## Day-of-Week Conditioning

The standard cGAN uses a five-dimensional one-hot representation:

```text
Monday
Tuesday
Wednesday
Thursday
Friday
```

This allows the model to test whether recurring calendar structure provides additional predictive information.

---

## Volatility-Regime Conditioning

A second conditional model represents market behaviour using three volatility states:

```text
low volatility
medium volatility
high volatility
```

The implementation derives regimes from rolling return volatility and divides the volatility distribution using quantile thresholds.

The resulting regime is represented using a three-dimensional one-hot condition vector.

This experiment investigates whether explicitly supplying market-state information improves the forecasting model.

---

## Stabilized cGAN

GAN training is often sensitive to optimization instability.

The repository therefore includes a stabilized conditional configuration using:

* multiple discriminator updates;
* label smoothing;
* noise injection;
* increased supervised prediction weight;
* gradient clipping.

The purpose is to test whether a more stable adversarial training process improves predictive performance.

---

## Ensemble Forecasting

The repository also evaluates simple prediction-level ensembles.

The primary ensemble averages:

$$
\hat{y}_{ensemble}
=
\frac{1}{2}
\left(
\hat{y}_{LSTM}
+
\hat{y}_{cGAN}
\right)
$$

A generalized multi-model evaluator also supports multiple LSTM and cGAN checkpoints with configurable weights.

This provides a simple way to examine whether combining models reduces individual-model error.

---

## Evaluation

Predictions are evaluated after conversion back to USD.

The repository uses:

### Mean Absolute Error

$$
MAE =
\frac{1}{n}
\sum_{i=1}^{n}
|y_i-\hat{y}_i|
$$

### Root Mean Squared Error

$$
RMSE =
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
}
$$

### Mean Absolute Percentage Error

$$
MAPE =
\frac{100}{n}
\sum_{i=1}^{n}
\left|
\frac{y_i-\hat{y}_i}{y_i}
\right|
$$

---

## Saved Test Results

The repository includes saved predictions for **120 test observations from 13 July 2020 to 30 December 2020**:

```text
reports/results/window_predictions_test.csv
```

Metrics calculated from that stored artifact are:

| Model                |  MAE (USD) | RMSE (USD) |       MAPE |
| -------------------- | ---------: | ---------: | ---------: |
| **LSTM**             | **2.0967** | **2.8316** | **1.034%** |
| GAN                  |     2.1093 |     2.8648 |     1.041% |
| cGAN — DOW           |     2.1312 |     2.8753 |     1.051% |
| cGAN — Regime        |     2.1176 |     2.8987 |     1.045% |
| Stable cGAN — DOW    |     2.1323 |     2.9511 |     1.054% |
| LSTM + cGAN Ensemble |     2.1122 |     2.8509 |     1.042% |
| Ensemble++           |     2.1056 |     2.8675 |     1.040% |

### Interpretation

The most important result is that **the simpler LSTM baseline slightly outperformed the GAN-based variants on the saved test window**.

This is useful rather than disappointing.

It demonstrates that additional architectural complexity does not automatically translate into better forecasting performance.

The GAN and cGAN variants remain close to the baseline, but the stored results do not provide evidence that adversarial or conditional modelling produces a clear predictive advantage for this experiment.

---

## Visual Evaluation

The repository includes:

```text
reports/figures/compare_curves_test.png
reports/figures/errors_hist_test.png
```

These visualize:

* actual vs predicted closing prices;
* model-to-model forecast differences;
* prediction-error distributions.

The associated prediction values are retained in:

```text
reports/results/window_predictions_test.csv
```

---

## Repository Structure

```text
.
├── configs/
│   ├── dataset.yaml
│   ├── lstm_baseline.yaml
│   ├── gan_classical.yaml
│   ├── cgan.yaml
│   ├── cgan_stable.yaml
│   └── cgan_regime.yaml
│
├── src/
│   ├── data/
│   │   ├── ingest.py
│   │   ├── normalize.py
│   │   └── window.py
│   │
│   ├── models/
│   │   ├── lstm_reg.py
│   │   ├── gans_classical.py
│   │   └── cgan.py
│   │
│   ├── train/
│   │   ├── train_lstm.py
│   │   ├── train_gan.py
│   │   ├── train_cgan.py
│   │   ├── train_cgan_stable.py
│   │   ├── train_cgan_regime.py
│   │   └── train_ensemble.py
│   │
│   ├── eval/
│   │   ├── metrics.py
│   │   ├── evaluate_lstm.py
│   │   ├── evaluate_gan.py
│   │   ├── evaluate_cgan.py
│   │   ├── evaluate_cgan_stable.py
│   │   ├── evaluate_cgan_regime.py
│   │   ├── evaluate_ensemble.py
│   │   └── evaluate_ensemble_multi.py
│   │
│   └── viz/
│       └── plot_predictions.py
│
├── reports/
│   ├── figures/
│   │   ├── compare_curves_test.png
│   │   └── errors_hist_test.png
│   └── results/
│       └── window_predictions_test.csv
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Technology Stack

| Component            | Technology                   |
| -------------------- | ---------------------------- |
| Language             | Python                       |
| Deep learning        | PyTorch                      |
| Recurrent modelling  | LSTM                         |
| Generative modelling | GAN / Conditional GAN        |
| Data source          | Yahoo Finance via `yfinance` |
| Data processing      | Pandas / NumPy               |
| Configuration        | YAML                         |
| Evaluation           | MAE / RMSE / MAPE            |
| Visualization        | Matplotlib                   |

---

## Reproducing the Pipeline

Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 1. Download market data

```powershell
python -m src.data.ingest --config configs/dataset.yaml
```

### 2. Normalize the features

```powershell
python -m src.data.normalize --config configs/dataset.yaml
```

### 3. Build chronological windows

```powershell
python -m src.data.window --config configs/dataset.yaml
```

### 4. Train the LSTM baseline

```powershell
python -m src.train.train_lstm --config configs/lstm_baseline.yaml
```

### 5. Train the classical GAN

```powershell
python -m src.train.train_gan --config configs/gan_classical.yaml
```

### 6. Train the day-of-week cGAN

```powershell
python -m src.train.train_cgan --config configs/cgan.yaml
```

### 7. Train the stabilized cGAN

```powershell
python -m src.train.train_cgan_stable --config configs/cgan_stable.yaml
```

### 8. Train the regime-conditioned cGAN

```powershell
python -m src.train.train_cgan_regime --config configs/cgan_regime.yaml
```

Model checkpoints are written under:

```text
runs/
```

which is intentionally excluded from version control.

---

## Limitations

This repository is an experimental forecasting implementation rather than a trading system.

Several limitations are important when interpreting the results.

The experiment currently evaluates a single equity, MSFT, over one historical period. The holdout set is relatively small, and the results should not be generalized to other securities or market regimes without additional testing.

The current regime implementation derives volatility thresholds using the available historical series. A stricter forecasting evaluation should estimate regime thresholds using training data only.

The moving-average normalization and denormalization pipeline is designed for this experimental workflow. A production forecasting pipeline should ensure that all normalization statistics used at prediction time are strictly causal and available without reference to the future target observation.

The project also evaluates point-forecast accuracy rather than profitability, transaction costs, risk-adjusted return, or economic utility.

---

## Future Work

Natural extensions include walk-forward validation, strictly causal normalization, training-only regime estimation, multiple equities, market-index and macroeconomic conditioning, probabilistic forecasting, richer adversarial objectives, attention-based temporal models, and statistical significance testing across repeated runs.

---

## Research Takeaway

The strongest finding from this project is methodological:

> More complex generative models should be treated as hypotheses to test, not assumptions of superior performance.

In the stored experiment, the supervised LSTM baseline remained slightly stronger than the GAN and conditional-GAN variants.

The value of the project is therefore not simply in implementing GANs, but in constructing a common pipeline where alternative forecasting architectures can be compared under the same data and evaluation framework.

---

## Disclaimer

This repository is intended for educational and research purposes.

The forecasts are not investment advice and should not be interpreted as recommendations to buy or sell financial assets.

---

## License

Original source code and documentation in this repository are released under the **MIT License**.

Third-party libraries, market data, and referenced research remain subject to their respective licenses and terms.

---

## Author

**Suha Roy**

M.Sc. Computer Science — Big Data Analytics
Central University of Rajasthan

GitHub: https://github.com/suharoy
