
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split


def classify_packets(df_packets, model):
    # separate to 2 data frames of features and prediction
    features_name = ["b_{}".format(bit) for bit in range(0, 64)]
    features = df_packets[features_name]
    prediction = df_packets.rule

    print('Split the data')
    train_fet, test_fet, train_val, test_val = train_test_split(features, prediction, test_size=0.2, random_state=0)
    print('fit model')
    model.fit(train_fet, train_val)
    print('predict')
    prediction_value = model.predict(test_fet)
    print('Success rate: {}'.format(accuracy_score(test_val, prediction_value)))
    print('MSE: {}'.format(mean_absolute_error(test_val, prediction_value)))
