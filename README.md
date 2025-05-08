# 🎬 Movie Revenue Prediction

Predicting movie box office revenue using metadata from the [TMDB](https://www.themoviedb.org/) dataset with feature engineering, exploratory data analysis, and machine learning models including Linear Regression and Random Forest.

---

## 📂 Project Structure

```
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── notebooks/
│   └── movie_revenue_prediction.ipynb
├── models/
│   ├── linear_model.pkl
│   └── random_forest_model.pkl
├── submission.csv
└── README.md
```

---

## 🚀 Features

* 🔍 **EDA & Cleaning**: Genres, keywords, cast, language trends, release date distributions.
* ⚙️ **Feature Engineering**: Parsed JSON fields, log-transformed features, profit calculation.
* 📊 **Visualization**: Revenue/budget over time, keyword frequency, genre profitability.
* 🤖 **Modeling**:

  * Multi-feature **Linear Regression**
  * **Random Forest Regressor** with hyperparameter tuning (`GridSearchCV`)
* 📈 **Evaluation**: R² score, RMSE, and prediction visualization.
* 📤 **Submission**: Generates `submission.csv` in the required format.
* 💾 **Model Export**: Trained models saved using `joblib`.

---

## 🧠 Tech Stack

* Python (Pandas, NumPy, Matplotlib, Seaborn)
* scikit-learn
* joblib
* Jupyter Notebook

---

## 🏁 Getting Started

1. **Clone this repo**

   ```bash
   git clone https://github.com/yourusername/movie-revenue-prediction.git
   cd movie-revenue-prediction
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   Place `train.csv`, `test.csv`, and `sample_submission.csv` in the `data/` folder.

4. **Run the notebook**
   Open `notebooks/movie_revenue_prediction.ipynb` and run all cells.

---

## 📁 Data Source

This project uses the [TMDB Box Office Prediction dataset](https://www.kaggle.com/c/tmdb-box-office-prediction/data) from Kaggle, which includes:

* Movie metadata (cast, crew, budget, revenue, etc.)
* JSON-encoded fields for genres, production, etc.

---

## 📊 Results

* Best **R² score**: \~0.72 (log-transformed revenue)
* RMSE reduced after using log transformations and Random Forest with tuning.

---

## 📌 Future Improvements

* Add support for ensemble models (XGBoost, LightGBM)
* Use NLP for `overview` or `tagline`
* Deploy as a web app with Streamlit or Flask

---
