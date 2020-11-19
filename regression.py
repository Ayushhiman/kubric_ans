import requests
import pandas as pd
import scipy
import numpy
import sys

TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    print(response)
    df=pd.read_csv(TRAIN_DATA_URL).T.reset_index()
    df.columns=['area','price']
    df=df.iloc[1:]
    df['area']=pd.to_numeric(df['area'],downcast='float')
    df['price']=pd.to_numeric(df['price'],downcast='float')
    X=df['area'].values
    Y=df['price'].values
    z=numpy.polyfit(X,Y,3)
    p=numpy.poly1d(z)
    predictions=p(area)
    return predictions
if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    print(areas.shape)
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
