# Employee Salary Prediction

Binary income classification on the UCI **Adult census** dataset — predicts whether
a person earns `>50K` or `<=50K` from demographic and employment features.

Five scikit-learn classifiers are trained and compared; the strongest is wrapped in a
pipeline, persisted with `joblib`, and served through a small Streamlit app.

## Approach

**Data cleaning**
- Replaced `?` placeholders in `workclass` and `occupation` with `NotListed`
- Dropped `Without-pay` and `Never-worked` classes (too few samples to learn from)
- Removed age and education outliers

**Features** — age, workclass, educational-num, marital-status, occupation,
relationship, race, gender, capital-gain, capital-loss, hours-per-week, native-country

**Models compared**

| Model |
|---|
| Logistic Regression |
| Random Forest |
| K-Nearest Neighbours |
| Support Vector Machine |
| Gradient Boosting |

Each is evaluated on accuracy and a full `classification_report`, with an 80/20
stratified split (`random_state=42`). The best performer is retrained inside a
`Pipeline` with `StandardScaler` and exported to `employee_salary_predictor.pkl`.

## Files

| File | Purpose |
|---|---|
| `employee_salary.ipynb` | Full analysis — EDA, cleaning, model comparison, export |
| `employee_salary.py` | Script version of the same pipeline |

## Running it

```bash
pip install pandas scikit-learn matplotlib joblib streamlit
jupyter notebook employee_salary.ipynb
```

The dataset (`adult.csv`) is from the
[UCI Adult repository](https://archive.ics.uci.edu/dataset/2/adult) and is not
committed here — download it and update the path in the first cell.

To run the Streamlit front-end after the model has been exported:

```bash
streamlit run app.py
```

## Notes

Class balance in this dataset is roughly 75/25, so accuracy alone overstates
performance — the `classification_report` precision and recall on the minority
`>50K` class are the numbers worth reading.

---

Built while learning applied ML during the AICTE / Edunet internship programme.
