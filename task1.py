import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'SquareFootage': [800, 1000, 1200, 1500, 1800, 2000],
    'Bedrooms': [1, 2, 2, 3, 3, 4],
    'Bathrooms': [1, 1, 2, 2, 3, 3],
    'Price': [3000000, 4000000, 5000000, 6500000, 8000000, 9000000]
}

df = pd.DataFrame(data)

X = df[['SquareFootage', 'Bedrooms', 'Bathrooms']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

new_house = [[1600, 3, 2]]
predicted_price = model.predict(new_house)

print("Predicted House Price:", predicted_price[0])