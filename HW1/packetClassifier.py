from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def classify_packets(df_packets):
    # separate to 2 data frames of features and prediction
    features_name = ["b_{}".format(bit) for bit in range(0, 64)]
    features = df_packets[features_name]
    prediction = df_packets.rule

    print('Split the data')
    train_fet, test_fet, train_pred, test_pred = train_test_split(features, prediction, test_size=0.2, random_state=0)

    print('fit model')
    forest_model = RandomForestRegressor(random_state=1)
    forest_model.fit(train_fet, train_pred)

    print('predict')
    prediction_value = forest_model.predict(test_fet)
    print(mean_absolute_error(test_pred, prediction_value))