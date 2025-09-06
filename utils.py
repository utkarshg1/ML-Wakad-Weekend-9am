import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator


def evaluate_single_model_class(
    model: BaseEstimator,
    xtrain: pd.DataFrame,
    ytrain: pd.Series,
    xtest: pd.DataFrame,
    ytest: pd.Series,
) -> dict:
    # Cross validate
    scores = cross_val_score(model, xtrain, ytrain, cv=5, scoring="f1_macro")
    cv_mean = scores.mean().round(4)
    cv_std = scores.std().round(4)
    # Fit the model
    model.fit(xtrain, ytrain)
    # Predict results for train and test
    ypred_train = model.predict(xtrain)
    ypred_test = model.predict(xtest)
    # Calculate f1 macro on train and test
    f1_train = round(f1_score(ytrain, ypred_train, average="macro"), 4)
    f1_test = round(f1_score(ytest, ypred_test, average="macro"), 4)
    gen_err = round(abs(f1_train - f1_test), 4)
    return {
        "model_name": type(model).__name__,
        "model": model,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "f1_train": f1_train,
        "f1_test": f1_test,
        "gen_err": gen_err,
    }


def algo_evaluation(
    models: list[BaseEstimator],
    xtrain: pd.DataFrame,
    ytrain: pd.Series,
    xtest: pd.DataFrame,
    ytest: pd.DataFrame,
) -> tuple[BaseEstimator, pd.DataFrame]:
    results = []
    # Apply for loop on all models
    for model in models:
        r = evaluate_single_model_class(model, xtrain, ytrain, xtest, ytest)
        print(r)
        results.append(r)

    # Convert dictionary to dataframe
    results_df = pd.DataFrame(results)

    # Select the model based on cv_mean
    sort_df = results_df.sort_values(by="cv_mean", ascending=False).reset_index(
        drop=True
    )
    best_model = sort_df.loc[0, "model"]
    return best_model, sort_df
