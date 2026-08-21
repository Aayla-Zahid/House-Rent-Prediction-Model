# 🏠 House Rent Prediction

A machine learning web app that predicts monthly house rent in Indian cities based on property details like size, number of bedrooms, city, furnishing status, and more. Built with **scikit-learn** for the model and **Streamlit** for the interactive interface.

## Demo

Enter details like BHK, size, city, and furnishing status, and the app instantly predicts an estimated monthly rent.

## Dataset

The model is trained on the [House Rent Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset), containing 4,746 rental listings across 6 major Indian cities: Kolkata, Mumbai, Bangalore, Delhi, Chennai, and Hyderabad.

**Features used:**

| Feature | Type | Description |
|---|---|---|
| `BHK` | Numeric | Number of bedrooms |
| `Size` | Numeric | Property size (sq ft) |
| `Bathroom` | Numeric | Number of bathrooms |
| `Floor` | Categorical | Floor number and total floors in the building (e.g. "2 out of 4") |
| `Area Type` | Categorical | Super Area / Carpet Area / Built Area |
| `Area Locality` | Categorical | Specific neighborhood/locality |
| `City` | Categorical | One of 6 major Indian cities |
| `Furnishing Status` | Categorical | Unfurnished / Semi-Furnished / Furnished |
| `Tenant Preferred` | Categorical | Bachelors / Family / Bachelors or Family |
| `Point of Contact` | Categorical | Owner / Agent / Builder |

**Target:** `Rent` (monthly rent in ₹)

## Model

- **Algorithm:** Linear Regression (scikit-learn)
- **Preprocessing:** Categorical columns encoded with `LabelEncoder` — a separate fitted encoder is saved per column (not shared) to preserve each column's own category-to-number mapping
- **Train/test split:** 80/20
- **Evaluation metric:** Mean Absolute Error (MAE) on the held-out test set

## Project Structure

```
├── Machine_Learning__R_.ipynb    # EDA, preprocessing, model training & evaluation
├── House_Rent_Dataset.csv        # Raw dataset
├── app.py                        # Streamlit web app
├── ml_Regression_model.pkl       # Trained Linear Regression model
├── ml_Label_Encoder.pkl          # Dict of fitted LabelEncoders (one per categorical column)
└── README.md
```

## How It Works

1. **Training** (`Machine_Learning__R_.ipynb`): loads and cleans the dataset, encodes categorical columns, trains a Linear Regression model, evaluates it with MAE, and saves the model + encoders with `joblib`.
2. **Inference** (`app.py`): collects property details through a Streamlit form, encodes the categorical inputs using the same saved encoders, and feeds the full feature vector into the trained model to produce a rent prediction.

## Getting Started

### Prerequisites

```bash
pip install streamlit scikit-learn pandas numpy joblib
```

### Run the app

Make sure `app.py`, `ml_Regression_model.pkl`, and `ml_Label_Encoder.pkl` are in the same folder, then run:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Retrain the model

Open `Machine_Learning__R_.ipynb` in Jupyter and run all cells. This regenerates `ml_Regression_model.pkl` and `ml_Label_Encoder.pkl`.

## Limitations

- `Area Locality` has over 2,000 unique values in the training data, many appearing only once — predictions for localities not seen during training fall back to a default value rather than a locality-specific estimate.
- Linear Regression assumes a linear relationship between features and rent; it may underperform on non-linear pricing patterns compared to tree-based models (e.g. Random Forest, XGBoost).

## License

This project is for educational purposes.
