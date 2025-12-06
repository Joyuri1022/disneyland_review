import joblib
import pandas as pd

model = joblib.load("../models/tfidf_logreg.joblib")

sample = pd.DataFrame([{
    "Review_Text_clean": "the ride was amazing and staff was friendly",
    "Branch": "California",
    "Reviewer_Location": "United States",
    "Year": 2019,
    "Month": 5
}])

pred = model.predict(sample)[0]
print("Predicted rating:", pred)
