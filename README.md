# Stock Price Direction Predictor with Sentiment Analysis

A machine learning project that predicts whether stock prices will go up or down by combining technical indicators (like RSI, MACD, SMA) with sentiment analysis from financial news headlines. Built using an XGBoost classifier and visualized through an interactive Streamlit dashboard, this tool offers a powerful blend of quantitative and qualitative market signals for directional prediction.

![image](https://github.com/user-attachments/assets/6e728c86-8e9e-487d-b2d0-e9614718de90)

---

## Project Features

- **Sentiment Analysis** with VADER on news headlines
- **Technical Indicator Calculation** (e.g., RSI, MACD, SMA, etc.)
- **XGBoost Classifier** with optional hyperparameter tuning
- **Streamlit Dashboard** to visualize predictions and metrics
- Logging of both feature engineering and model training

---

## Project Structure

```

stock-market-predictor/
│
├── data/
│   ├── raw/                   # Raw CSVs like news headlines
│   └── processed/             # Feature-engineered datasets
│
├── logs/
│   ├── feature\_engineering.log
│   └── model\_training.log
│
├── src/
│   ├── sentiment.py           # VADER sentiment scoring
│   ├── data\_loader.py         # Loads historical price/news data
│   ├── features.py            # Adds technical indicators
│   └── train\_model.py         # Training and evaluation
│
├── app.py                     # Streamlit dashboard
├── Requirements.txt           # Python dependencies
└── README.md                  # You're here!

````

---

## Installation

1. Clone the repository:

  ```bash
   git clone https://github.com/yashgolani28/Stock-Price-Direction-Predictor-with-Sentiment-Analysis.git
   cd Stock Price Direction Predictor with Sentiment Analysis
  ```

2. Create a virtual environment (optional but recommended):

  ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```

3. Install dependencies:

  ```bash
   pip install -r Requirements.txt
  ```

---

## How to Run

1. Prepare your data:

   * Place news data in `data/raw/`
   * Run `sentiment.py` to generate sentiment scores
   * Run feature engineering scripts and store output in `data/processed/`

2. Train the model:

   ```bash
   python src/train_model.py
   ```

3. Launch the web app:

   ```bash
   streamlit run app.py
   ```

---

## Results

### Model Performance

![Screenshot 2025-05-13 130334](https://github.com/user-attachments/assets/b6e71eb6-d509-42db-a611-a94bd5590fd0)
![Screenshot 2025-05-13 130633](https://github.com/user-attachments/assets/7cb1de67-ca29-4a8d-b91a-0b042bc5b940)
![Screenshot 2025-05-13 130657](https://github.com/user-attachments/assets/22e704a2-b2d4-4f32-b833-3922d1b0976f)

> *Results only for Apple*

---

### Feature Importance

![newplot](https://github.com/user-attachments/assets/2640a5ed-ed96-42ec-99ac-5cd0bf0f022a)

> *Results only for Apple*

---

### Streamlit Dashboard

![Screenshot 2025-05-13 131155](https://github.com/user-attachments/assets/47824f82-d9fd-4404-b736-727ca1531dde)
![image](https://github.com/user-attachments/assets/46d33c6b-7fe9-4a18-a467-cacfa0b5d4ac)
![image](https://github.com/user-attachments/assets/bf15f747-c18b-4250-aa12-3a9e59093fd8)
![image](https://github.com/user-attachments/assets/293a0a58-13dc-4a4a-8906-6c0abe169374)

> *Results only for Apple*

---

## Future Improvements

* Integrate real-time news feeds
* Use LSTM for sequential price movement prediction
* Add more financial indicators
* Improve UI with more interactive components

---

## Acknowledgements

* [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
* [TA-Lib](https://mrjbq7.github.io/ta-lib/)
* [Streamlit](https://streamlit.io/)
* [XGBoost](https://xgboost.readthedocs.io/en/latest/)



