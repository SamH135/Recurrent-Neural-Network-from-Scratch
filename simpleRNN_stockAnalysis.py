import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(filename='experiment_log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


# Load and preprocess data
def load_and_preprocess_data(file_path):
    # Load data
    df = pd.read_csv(file_path)

    # Convert the 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Handling any missing values
    df.fillna(method='ffill', inplace=True)

    # Separate 'Date' column
    dates = df['Date']

    # Select only numerical columns for analysis
    df_numerical = df.select_dtypes(include=['float64', 'int64'])

    return df_numerical, dates


def apply_pca(df, n_components=2):
    # Normalizing the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Applying PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)

    # Creating a DataFrame for principal components
    pc_df = pd.DataFrame(data=principal_components, columns=[f'PC{i + 1}' for i in range(n_components)])

    return pc_df, pca


# Load and preprocess datasets
infy_df, infy_dates = load_and_preprocess_data(
    r'https://raw.githubusercontent.com/SamH135/National-Stock-Exchange-Data/main/archive/infy_stock.csv')
nifty_it_df, nifty_it_dates = load_and_preprocess_data(
    r'https://raw.githubusercontent.com/SamH135/National-Stock-Exchange-Data/main/archive/nifty_it_index.csv')
tcs_df, tcs_dates = load_and_preprocess_data(
    r'https://raw.githubusercontent.com/SamH135/National-Stock-Exchange-Data/main/archive/tcs_stock.csv')

# Apply PCA to each preprocessed dataset
infy_pca_df, infy_pca = apply_pca(infy_df)
nifty_it_pca_df, nifty_it_pca = apply_pca(nifty_it_df)
tcs_pca_df, tcs_pca = apply_pca(tcs_df)

logging.info("Start of log")


# Visualize PCA explained variance
def plot_pca_explained_variance(pca, title):
    plt.figure(figsize=(10, 7))
    plt.bar(range(1, pca.n_components_ + 1), pca.explained_variance_ratio_, alpha=0.5, align='center')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.title(title)
    plt.show()


# Visualize PCA explained variance for each dataset
plot_pca_explained_variance(infy_pca, 'PCA Explained Variance Ratio for INFOSYS Dataset')
plot_pca_explained_variance(nifty_it_pca, 'PCA Explained Variance Ratio for NIFTY_IT_INDEX Dataset')
plot_pca_explained_variance(tcs_pca, 'PCA Explained Variance Ratio for TCS Dataset')


# Function for plotting stock prices and volume
def plot_stock_trends(df, dates, title):
    # Merge the dates with df for plotting
    df_with_dates = pd.concat([dates.reset_index(drop=True), df.reset_index(drop=True)], axis=1)

    # Plotting stock prices
    df_with_dates.plot(x='Date', y=['Open', 'High', 'Low', 'Close'], figsize=(10, 6))
    plt.title(f'{title} - Stock Prices Over Time')
    plt.ylabel('Price')
    plt.show()

    # Plotting stock volume
    df_with_dates.plot(x='Date', y='Volume', figsize=(10, 6))
    plt.title(f'{title} - Trading Volume Over Time')
    plt.ylabel('Volume')
    plt.show()


# Function for generating statistical summaries
def generate_statistics(df):
    # Check if 'Date' column exists in the DataFrame and drop it if present
    if 'Date' in df.columns:
        df = df.drop(columns=['Date'])

    # Generate statistical summary
    stats = df.describe()
    return stats


# Plot stock trends for INFOSYS
plot_stock_trends(infy_df, infy_dates, 'INFOSYS')

# Generate and print statistical summaries for INFOSYS
infy_stats = generate_statistics(infy_df)
print(infy_stats)


class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
        self.learning_rate = learning_rate

    def forward(self, inputs):
        h = np.zeros((self.Wxh.shape[0], 1))
        self.last_inputs = inputs
        self.last_hs = {0: h}

        for i, x in enumerate(inputs):
            # Ensure x is a column vector
            x = np.array([x]).reshape(-1, 1)

            # Compute the new hidden state
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)

            # Store the hidden state with correct shape
            self.last_hs[i + 1] = h

        y = np.dot(self.Why, h) + self.by
        return y, h

    def backward(self, d_y, learn=True):
        n = len(self.last_inputs)

        # Ensure d_y is 2D with shape (output_size, 1)
        if d_y.ndim == 3:
            d_y = d_y.reshape(d_y.shape[1], d_y.shape[2])

        d_Why = np.dot(d_y, self.last_hs[n].T)
        d_by = d_y

        d_Whh = np.zeros_like(self.Whh)
        d_Wxh = np.zeros_like(self.Wxh)
        d_bh = np.zeros_like(self.bh)

        d_h = np.dot(self.Why.T, d_y)

        for t in reversed(range(n)):
            temp = ((1 - self.last_hs[t + 1] ** 2) * d_h)
            d_bh += temp
            input_t = np.array([self.last_inputs[t]]).reshape(-1, 1)  # Reshape input to column vector
            d_Wxh += np.dot(temp, input_t.T)
            d_Whh += np.dot(temp, self.last_hs[t].T)

            d_h = np.dot(self.Whh.T, temp)

        # Update weights if learning is True
        if learn:
            self.Why -= self.learning_rate * d_Why
            self.Whh -= self.learning_rate * d_Whh
            self.Wxh -= self.learning_rate * d_Wxh
            self.bh -= self.learning_rate * d_bh
            self.by -= self.learning_rate * d_by

    def loss(self, y_pred, y_true):
        return ((y_pred - y_true) ** 2).mean()


# Calculate evaluation metrics
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def r_squared(y_true, y_pred):
    return r2_score(y_true, y_pred)


# Use Infosys closing price for prediction
data = infy_df['Close'].values

# Normalize the data
data_normalized = StandardScaler().fit_transform(data.reshape(-1, 1)).reshape(-1)

# Create sequences for training
sequence_length = 10
sequences = [data_normalized[i:i + sequence_length] for i in range(len(data_normalized) - sequence_length)]

# Initialize and train the RNN
input_size = 1  # Single feature
hidden_size = 40  # Number of hidden units
output_size = 1  # Single output

# Set training and testing sizes
train_size = int(len(sequences) * 0.8)
test_size = len(sequences) - train_size

# Split sequences into training and testing sets
train_sequences = sequences[:train_size]
test_sequences = sequences[train_size:]

# Log the train/test split and dataset size
logging.info(f"Train/Test Split = 80:20, Size of dataset = {len(data_normalized)}")

rnn = SimpleRNN(input_size, hidden_size, output_size)
logging.info(f"RNN parameters: input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}")
logging.info(f"RNN structure: Number of layers = 1, Neurons in hidden layer = {hidden_size}")
logging.info("Error Function used during training: Mean Squared Error (MSE)")
logging.info("Regularization: None")

# Training loop
epoch_list = []
loss_list = []
for epoch in range(50):
    epoch_list.append(epoch)
    total_loss = 0
    for seq in train_sequences:
        inputs = np.array(seq[:-1]).reshape(-1, 1)
        target = np.array([[seq[-1]]])  # Target shape is (1, 1)
        output, _ = rnn.forward(inputs)
        loss = rnn.loss(output, target)
        total_loss += loss
        d_y = 2 * (output - target)  # d_y will be of shape (1, 1)
        rnn.backward(d_y)

    # Calculate and log the training RMSE after each epoch
    train_rmse = np.sqrt(total_loss / len(train_sequences))
    logging.info(f"Epoch {epoch} completed with total loss: {total_loss}, Training RMSE: {train_rmse}")

    loss_list.append(total_loss)

plt.plot(epoch_list, loss_list, marker='o', linestyle='-', color='blue')
plt.title('Epoch vs Total Loss')
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.grid(True)
plt.show()

# Predict using the trained RNN on test data
test_predicted_prices = []
for seq in test_sequences:
    inputs = np.array(seq[:-1]).reshape(-1, 1)
    output, _ = rnn.forward(inputs)
    test_predicted_prices.append(output[0, 0])

# Denormalize the predicted prices for test data
test_predicted_prices = StandardScaler().fit(data.reshape(-1, 1)).inverse_transform(
    np.array(test_predicted_prices).reshape(-1, 1))

# Calculate metrics on test data
actual_prices_test = data[train_size + sequence_length:]
rmse_test = rmse(actual_prices_test, test_predicted_prices)
mae_test = mae(actual_prices_test, test_predicted_prices)
r2 = r_squared(actual_prices_test, test_predicted_prices)

# Log the performance on test data
logging.info(f"Test RMSE: {rmse_test}, Test MAE: {mae_test}, Test R^2: {r2}")

# Extract the actual prices for the test data
actual_prices_test = data[train_size + sequence_length:]

# Plot the actual and predicted prices for the test data
plt.plot(actual_prices_test, label='Actual Data')
plt.plot(test_predicted_prices, label='Predicted Prices')
plt.title('Test Data: Actual vs Predicted Prices')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.legend()
plt.show()

# Log the performance metrics for test data
logging.info(f"Test RMSE: {rmse_test:.4f}, Test MAE: {mae_test:.4f}, Test R^2: {r2}")

print("RMSE:", rmse_test)
print("MAE:", mae_test)
print("Test R^2:", r2)

# Print the shape of PCA components and number of columns after dropping 'Date'
print("PCA Components shape:", infy_pca.components_.shape)
print("Number of columns:", len(infy_df.columns))


def pca_feature_importance(pca, columns):
    components = np.abs(pca.components_)
    feature_importance = pd.DataFrame(components, columns=columns[:components.shape[1]])
    feature_importance.index = [f'PC{i + 1}' for i in range(pca.n_components_)]
    return feature_importance


infy_pca_df, infy_pca = apply_pca(infy_df)
infy_feature_importance = pca_feature_importance(infy_pca, infy_df.columns)
print("Feature Importance for INFOSYS PCA:")
print(infy_feature_importance)
