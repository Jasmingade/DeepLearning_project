# preprocess.py
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    # Assuming 'data' is a pandas DataFrame and the first column is the sample ID or similar
    features = data.iloc[:, 1:]  # Exclude sample ID if present

    # Normalize features for VAE
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Return the processed features and scaler for inverse transformation later
    return scaled_features, scaler
