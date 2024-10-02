from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor 
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import  StackingRegressor
import pandas as pd
import numpy as np
import joblib


# Đọc dữ liệu từ file csv
file_path = 'Weather Data.csv'
weather_data = pd.read_csv(file_path)
print(weather_data)

# Bỏ cột "Date/Time"
weather_data_cleaned = weather_data.drop(columns=['Date/Time'])

# Mã hóa "Weather" sang dạng số
label_encoder = LabelEncoder()
weather_data_cleaned['Weather'] = label_encoder.fit_transform(weather_data_cleaned['Weather'])
print(weather_data_cleaned)

# Fit the encoder on the "Weather" column
weather_data['Weather'] = label_encoder.fit_transform(weather_data['Weather'])

# Save the fitted LabelEncoder to a file
joblib.dump(label_encoder, 'label_encoder.pkl')

X = weather_data_cleaned.drop(columns=['Temp_C'])
y = weather_data_cleaned['Temp_C']

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra (80% huấn luyện, 20% kiểm tra)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Hiển thị dữ liệu đã được làm sạch và hình dạng của tập huấn luyện/kiểm tra
(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# Khởi tạo mô hình hồi quy tuyến tính
linear_model = LinearRegression()


# Huấn luyện mô hình
linear_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_linear = linear_model.predict(X_test)


# Đánh giá mô hình
mse_linear = mean_squared_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mse_linear)
mae_linear = mean_absolute_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)
print(f"Linear Regression MSE: {mse_linear}")
print(f"Linear Regression R-square: {r2_linear}\n")


# Giả sử mô hình tốt nhất là mô hình hồi quy tuyến tính
joblib.dump(linear_model, 'best_model.pkl')

# Khởi tạo mô hình Lasso
lasso_model = Lasso(alpha=0.01)

# Huấn luyện mô hình
lasso_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_lasso = lasso_model.predict(X_test)

# Đánh giá mô hình
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mse_lasso)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)
print(f"Lasso Regression MSE: {mse_lasso}")
print(f"Lasso Regression R-square: {r2_lasso}\n")

# Mô hình lasso ra file pkl
joblib.dump(lasso_model, 'best_model_lasso.pkl')

# Khởi tạo MLPRegressor với các tham số phù hợp
mlp_model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000,  early_stopping=True, validation_fraction=0.1, random_state=42)

# Huấn luyện mô hình
mlp_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_mlp = mlp_model.predict(X_test)

# Đánh giá mô hình
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
rmse_mlp = np.sqrt(mse_mlp)
mae_mlp = mean_absolute_error(y_test, y_pred_mlp)
r2_mlp = r2_score(y_test, y_pred_mlp)
print(f"MLPRegressor MSE: {mse_mlp}")
print(f"Lasso Regression R-square: {r2_mlp}\n")


#
joblib.dump(mlp_model, 'best_model_MLP.pkl')


# Sử dụng Stacking với các mô hình Linear Regression, Lasso, và MLPRegressor
estimators = [
    ('linear', LinearRegression()),
    ('lasso', Lasso(alpha=0.1)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000,  early_stopping=True, validation_fraction=0.1, random_state=42))
]

stacking_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())

# Huấn luyện mô hình Stacking
stacking_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra bằng Stacking
y_pred_stacking = stacking_model.predict(X_test)

# Đánh giá mô hình Stacking
mse_stacking = mean_squared_error(y_test, y_pred_stacking)
rmse_stacking = np.sqrt(mse_stacking)
mae_stacking = mean_absolute_error(y_test, y_pred_stacking)
r2_stacking = r2_score(y_test, y_pred_stacking)

print(f"Stacking Regression MSE: {mse_stacking}")
print(f"Stacking Regression RMSE: {rmse_stacking}")
print(f"Stacking Regression MAE: {mae_stacking}")
print(f"Stacking Regression R-Square: {r2_stacking}")

joblib.dump(stacking_model, 'best_model_Stacking.pkl')
