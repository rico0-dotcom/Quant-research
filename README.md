# Quant-research
# Bitcoin Price Direction Prediction with XGBoost

This repository contains code and sample data for a machine learning model that predicts Bitcoin’s 30-day price direction using on-chain, social, and macroeconomic indicators.

---

## Contents

- `btc_insight` — Main Python script with data processing, model training, hyperparameter tuning (500 Optuna trials), and SHAP explainability.  
- `sample.csv` — Sample of the original dataset (last 10 rows) to illustrate data format.  
- `optuna_output_log.txt` — Log file showing the hyperparameter tuning process and best trial results.

---

## Dataset

The full dataset used for training spans from **August 2022 to 2025**, as exchange reserves data was available starting August 2022. The features include on-chain metrics (NUPL, SOPR), social sentiment, miner flows, macroeconomic indicators (DXY, CPI, Fed Rate), and exchange reserves.

**Note:** For privacy and licensing reasons, the full dataset is not included here. The sample CSV demonstrates expected data structure.

---

## How to Run

1. Clone this repository:
    ```bash
    git clone <[(https://github.com/rico0-dotcom/Quant-research)]>
    cd <repo-folder>
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. For dataset use `sample.csv` for a quick demo (results will not be meaningful with only 10 rows).

4. Run the script:
    

---

## Hyperparameter Tuning

- Hyperparameter optimization is performed using [Optuna](https://optuna.org/) with **500 trials** to find the best XGBoost model parameters.  
- Time-series cross-validation (`TimeSeriesSplit` with 5 folds) is used to prevent look-ahead bias.  
- Logs of the tuning process and best parameters are saved in `optuna_output_log.txt`.

---

## Results & Insights

- The final model achieved **62.1% accuracy** with **91% recall on upward Bitcoin movements**.  
- SHAP analysis revealed some counterintuitive relationships between Bitcoin price direction and macro factors like the USD Index (DXY) and CPI inflation surprises.  
- See the LinkedIn post for a detailed discussion: [LinkedIn Post Link]

---

## Disclaimer

This project is for educational and research purposes only and is **not financial advice**. Trading cryptocurrencies carries significant risk.

---

## Contact

For questions or collaboration, feel free to reach out via LinkedIn: [Your LinkedIn Profile URL]

---

