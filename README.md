# Student Exam Performance Predictor

A beginner-friendly, end-to-end machine learning project that predicts a student's **math score** from their gender, ethnicity, parental education level, lunch type, test-prep status, reading score, and writing score — served through a small Flask web app.

This README explains what every file does and how to run the project, written for someone new to ML projects.

---

## 1. What this project actually does

There are two separate "halves" to this project:

1. **Training half (offline, run once)** — reads a CSV of past student exam data, cleans it, trains several machine learning models, picks the best one, and saves it to disk as a `.pkl` file.
2. **Serving half (runs every time someone visits the website)** — loads that saved model, takes new input from a web form, and returns a predicted math score.

```
Raw data  --->  Clean & train  --->  Save best model  --->  Web app loads model  --->  Predicts for new students
```

---

## 2. Project structure

```
ml-project/
├── app.py                          # Flask web app (local dev entry point)
├── application.py                  # Same app, entry point used for deployment (e.g. AWS)
├── requirements.txt                # Python packages needed
├── setup.py                        # Makes src/ pip-installable
├── artifact/                       # Generated files land here (created automatically)
│   ├── raw.csv                     # Unmodified copy of the source data
│   ├── train.csv / test.csv        # 80/20 train-test split
│   ├── preprocessor.pkl            # Saved data-cleaning/encoding pipeline
│   └── model.pkl                   # Saved best-performing trained model
├── notebook/
│   ├── data/stud.csv                # Original raw dataset
│   ├── 1 EDA.ipynb                 # Exploratory data analysis notebook
│   └── 2 MODEL.ipynb               # Model prototyping notebook
├── templates/
│   ├── index.html                  # Simple landing page
│   └── home.html                   # The prediction form + result display
└── src/
    ├── logger.py                   # Logging setup
    ├── exception.py                # Custom error handling
    ├── util.py                     # Shared helper functions (save/load/evaluate)
    ├── components/
    │   ├── data_ingestion.py       # Step 1: load + split data
    │   ├── data_transformation.py  # Step 2: clean + encode data
    │   └── model_trainer.py        # Step 3: train + select best model
    └── pipeline/
        └── predict_pipeline.py     # Used by app.py to make a live prediction
```

---

## 3. How training works (step by step)

Run with:
```bash
python src/components/data_ingestion.py
```

This single command chains three stages together:

**Step 1 — `data_ingestion.py`**
- Reads `notebook/data/stud.csv` into a pandas DataFrame.
- Saves an unmodified copy as `artifact/raw.csv`.
- Splits the data 80% train / 20% test using `train_test_split`.
- Saves those as `artifact/train.csv` and `artifact/test.csv`.

**Step 2 — `data_transformation.py`**
- Numeric columns (`reading_score`, `writing_score`): missing values filled with the median, then scaled with `StandardScaler`.
- Categorical columns (gender, ethnicity, parental education, lunch, test prep): missing values filled with the most frequent value, one-hot encoded, then scaled.
- These are combined into a single `ColumnTransformer` called the **preprocessor**.
- The preprocessor is *fit* on the training data and *applied* to both train and test data, then saved to `artifact/preprocessor.pkl` so the exact same transformation can be reused later on new data.

**Step 3 — `model_trainer.py`**
- Trains and hyperparameter-tunes 8 different regression models: Random Forest, Decision Tree, Gradient Boosting, Linear Regression, KNN, XGBoost, CatBoost, and AdaBoost.
- Scores each with R² on the test set (via `GridSearchCV` in `src/util.py`'s `evaluate_models()`).
- Keeps whichever model scored highest, and refuses to save anything scoring below 0.6 R².
- Saves the winning model to `artifact/model.pkl`.

---

## 4. How the web app works (step by step)

Run with:
```bash
python app.py
```
Then open `http://localhost:5000` in your browser.

1. Visiting `/predictdata` (GET) shows the input form defined in `templates/home.html`.
2. Submitting the form (POST) hits `predict_datapoint()` in `app.py`, which reads all 7 form fields.
3. Those fields are wrapped into a `CustomData` object (`src/pipeline/predict_pipeline.py`), which converts them into a single-row pandas DataFrame with the same column names used during training.
4. `PredictPipeline.predict()` loads `model.pkl` and `preprocessor.pkl` from `artifact/`, transforms the new row through the *same* preprocessor used in training, then feeds it to the model.
5. The predicted math score is passed back into `home.html` and displayed to the user.

---

## 5. File-by-file explanation (for beginners)

| File | What it does |
|---|---|
| `src/logger.py` | Creates a timestamped log file under `logs/` and writes progress messages to it. Doesn't affect predictions — just useful for debugging. |
| `src/exception.py` | Defines `CustomException`, which wraps Python errors with the exact file name and line number where they happened. Used everywhere with `try/except`. |
| `src/util.py` | Three helpers: `save_object()` (pickle/save an object), `load_object()` (unpickle/load it back), `evaluate_models()` (grid-search, fit, and score a dict of models). |
| `src/components/data_ingestion.py` | Loads the raw CSV, splits it into train/test sets, saves all three to `artifact/`. |
| `src/components/data_transformation.py` | Builds and applies the cleaning/encoding pipeline (imputation, scaling, one-hot encoding); saves it as `preprocessor.pkl`. |
| `src/components/model_trainer.py` | Trains 8 candidate models, tunes their hyperparameters, picks the best by R² score, saves it as `model.pkl`. |
| `src/pipeline/predict_pipeline.py` | `CustomData` turns form inputs into a DataFrame; `PredictPipeline` loads the saved model + preprocessor and returns a prediction. |
| `app.py` | Flask app used for local development. Defines the `/` and `/predictdata` routes. |
| `application.py` | Identical to `app.py`, but this is the entry point some cloud platforms (like AWS Elastic Beanstalk) expect to find. |
| `templates/index.html` | A minimal landing page. |
| `templates/home.html` | The HTML form for entering a student's details, and where the predicted score is shown. |
| `setup.py` | Lets you `pip install -e .` so `src` can be imported as a package. |
| `requirements.txt` | List of Python packages the project depends on. |
| `notebook/1 EDA.ipynb` | Exploratory analysis of the raw dataset (not needed to run the app). |
| `notebook/2 MODEL.ipynb` | Early prototyping of the modeling approach (not needed to run the app). |

---

## 6. Setup & installation

```bash
# 1. Clone the repository
git clone https://github.com/ManishaDutt11/ml-project.git
cd ml-project

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (creates artifact/*.pkl and artifact/*.csv)
python src/components/data_ingestion.py

# 5. Run the web app
python app.py
```

Then visit `http://localhost:5000/predictdata` in your browser, fill in the form, and submit to see a predicted math score.

---

## 7. Known issues (worth knowing as a beginner)

- **Hardcoded Windows-style file paths.** `data_ingestion.py` uses `'notebook\data\stud.csv'` and `predict_pipeline.py` uses `'artifact\model.pkl'` / `'artifact\preprocessor.pkl'`. Backslashes are Windows path separators — this will likely fail on macOS/Linux. Fix by using forward slashes or `os.path.join(...)` instead, e.g. `os.path.join('notebook', 'data', 'stud.csv')`.
- **`app.py` runs with `debug=True`.** Fine for local development, but should be turned off before deploying anywhere public.
- **No input validation beyond the HTML form.** The Flask route trusts that `reading_score`/`writing_score` are valid numbers; malformed input will raise an unhandled error.

---

## 8. Tech stack

- **Language:** Python
- **Data handling:** pandas, numpy
- **ML models:** scikit-learn, XGBoost, CatBoost
- **Web framework:** Flask
- **Serialization:** dill (pickle-like)
