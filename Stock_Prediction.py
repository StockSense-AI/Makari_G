# Import necessary libraries
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import datetime

# Step 6: Visualize Results
def visualize_results(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Actual Prices', color='blue')
    plt.plot(y_pred, label='Predicted Prices', color='red', linestyle='--')
    plt.legend()
    plt.title('Actual vs Predicted Stock Prices')
    plt.show()

# Main Function
if __name__ == "__main__":
    # Parameters
    STOCK = str(input('Select a stock symbol: '))  # Choose a stock symbol
    START_DATE = '2020-01-01'
    today = datetime.date.today()
    END_DATE = today - datetime.timedelta(days=1)

    # Step 1: Fetch Data
    stock_data = fetch_stock_data(STOCK, START_DATE, END_DATE)
    print("Sample Data:")
    print(stock_data.head())

    # Step 2: Preprocess Data
    stock_data = preprocess_data(stock_data)

    # Step 3: Split Data
    X_train, X_test, y_train, y_test = split_data(stock_data)

    # Step 4: Train Model
    model = train_model(X_train, y_train)
    print("Model Trained!")
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)

    # Step 5: Evaluate Model
    y_pred, mae, rmse, r2 = evaluate_model(model, X_test, y_test)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

    # Step 6: Visualize Results
    visualize_results(y_test, y_pred)

