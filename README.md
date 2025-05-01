# NBA Points Prediction Using Linear Regression

This project predicts an NBA player's total **points (PTS)** for the 2024-25 season based on their stats using **Linear Regression**.

## Project Files

- **`linear_regression_manual.py`**  
  Implements linear regression **from scratch** using NumPy. This version calculates the cost function, gradients, and performs gradient descent manually to train the model.

- **`linear_regression_sklearn.py`**  
  Uses **scikit-learn's** `LinearRegression` model to predict total points. This version simplifies training and prediction using the `fit()` and `predict()` methods.

Both scripts retrieve live NBA player data using `nba_api`.

---

## Objective

Build, compare, and learn from two implementations of linear regression:
- A manual implementation using NumPy
- A library-based solution using `sklearn`

This provides both **theoretical understanding** and **practical experience** with machine learning.

---

## Features Used for Prediction

The model uses the following features to predict total points:
- AST (Assists)
- REB (Rebounds)
- FG_PCT (Field Goal %)
- GP (Games Played)
- PLUS_MINUS_RANK
- AGE
- STL (Steals)
- BLK (Blocks)
- TOV (Turnovers)
- FG3_PCT (3-Point %)
- FT_PCT (Free Throw %)

---

## How to Run

### 1. Install Requirements

```bash
pip install nba_api pandas numpy scikit-learn
