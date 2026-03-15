📊 HR Attrition Analysis & Prediction

1 Giới thiệu

Dự án này thực hiện phân tích và dự đoán việc nhân viên nghỉ việc (Employee Attrition) bằng các kỹ thuật Data Mining và Machine Learning.

Bộ dữ liệu được sử dụng là HR Analytics Dataset, một bộ dữ liệu phổ biến trong các bài toán phân tích nhân sự.

Dự án kết hợp hai hướng tiếp cận:

- Predictive Modeling → Dự đoán nhân viên có khả năng nghỉ việc

- Association Rule Mining → Phân tích các nguyên nhân dẫn đến nghỉ việc

Mục tiêu của dự án:

Xây dựng mô hình dự đoán nhân viên có nguy cơ nghỉ việc và tìm ra các yếu tố ảnh hưởng đến quyết định nghỉ việc.

2. Dataset
HR Analytics Dataset

| Thuộc tính              | Giá trị                 |
| ----------------------- | ----------------------- |
| Số bản ghi              | ~1480                   |
| Số thuộc tính ban đầu   | 35                      |
| Số thuộc tính sau xử lý | ~54                     |
| Biến mục tiêu           | `Attrition`             |
| Giá trị                 | `0 = Stay`, `1 = Leave` |

Các nhóm thuộc tính chính:

- Thông tin cá nhân

- Thu nhập

- Công việc

- Mức độ hài lòng

- Thâm niên làm việc

Ví dụ một số thuộc tính:
```
Age
MonthlyIncome
JobRole
Department
WorkLifeBalance
YearsAtCompany
OverTime
```

Dataset có class imbalance:

- Nhân viên nghỉ việc chiếm tỷ lệ nhỏ

Vì vậy các metric đánh giá tập trung vào:

- Recall

- Precision

- F1-score

- PR-AUC

3. Pipeline dự án

Quy trình thực hiện:
```
Raw Data
   ↓
Data Preprocessing
   ↓
Association Rule Mining
   ↓
Rule Clustering
   ↓
Machine Learning Models
   ↓
Semi-supervised Learning
   ↓
Evaluation & Insights
```

Các nhiệm vụ chính:

- Tiền xử lý dữ liệu

- Khai phá luật kết hợp

- Phân cụm luật

- Huấn luyện mô hình dự đoán

- Mô phỏng kịch bản thiếu nhãn

4. Công nghệ sử dụng

| Nhóm                  | Công nghệ           |
| --------------------- | ------------------- |
| Ngôn ngữ lập trình    | Python              |
| Xử lý dữ liệu         | Pandas, NumPy       |
| Trực quan hóa         | Matplotlib, Seaborn |
| Machine Learning      | Scikit-learn        |
| Gradient Boosting     | XGBoost             |
| Rule Mining           | mlxtend             |
| Clustering            | KMeans              |
| Tự động chạy notebook | Papermill           |
| Môi trường notebook   | Jupyter             |

5. Cấu trúc thư mục
```
HR_Attrition_Project
│
├── data
│   ├── raw
│   │   └── HR_Analytics.csv
│   │
│   └── processed
│       └── hr_processed_ml.csv
│
├── notebooks
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_rule_mining.ipynb
│   ├── 03_rule_clustering.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_semi_supervised_learning.ipynb
│
├── results
│   ├── figures
│   ├── rules
│   └── models
│
├── reports
│   └── research_log.pdf
│
├── requirements.txt
├── app.py
├── run_pipeline.py
└── README.md
```

6. Cài đặt môi trường
Clone repository
```
git clone https://github.com/yourusername/hr-attrition-analysis.git

cd hr-attrition-analysis
```

Tạo môi trường ảo

Sử dụng Conda
```
conda create -n hr_attrition python=3.9
conda activate hr_attrition
```

Cài đặt thư viện
```
pip install -r requirements.txt
```

7. Chạy pipeline bằng Papermill

```Papermill``` cho phép tự động chạy các notebook.

```
python run_papermill.py
```

8. Các thành phần chính
8.1 Tiền xử lý dữ liệu

Các bước thực hiện:

- Xử lý giá trị thiếu

- Loại bỏ các cột không cần thiết

- Mã hóa biến phân loại

- Chuẩn hóa dữ liệu bằng ```StandardScaler```

Kết quả:

```hr_processed_ml.csv```

Dataset này được dùng cho toàn bộ các bước modeling.

8.2 Khai phá luật kết hợp

Các luật kết hợp được khai phá nhằm tìm ra các mẫu liên quan đến nghỉ việc.

Ví dụ một luật:

```
OverTime = Yes
AND
YearsAtCompany < 2
→ Attrition = Leave
```

Các chỉ số đánh giá luật:

- Support

- Confidence

- Lift

Sau đó chọn 100 luật tốt nhất để phân tích.

8.3 Phân cụm luật

Các luật được phân cụm để tìm ra những nhóm nguyên nhân nghỉ việc khác nhau.

Quy trình:

- Chuyển luật thành vector đặc trưng

- Áp dụng trọng số:

    ```weight = lift × confidence```

- Áp dụng thuật toán KMeans

- Đánh giá bằng Silhouette Score

Số cụm tối ưu:

```K = 7```

Mỗi cụm đại diện cho một nhóm nguyên nhân nghỉ việc.

8.4 Mô hình dự đoán

Hai mô hình được sử dụng:

XGBoost

Cấu hình:

```
n_estimators = 300
max_depth = 5
learning_rate = 0.05
subsample = 0.8
```

Ngưỡng dự đoán tối ưu:

```threshold ≈ 0.25```

Kết quả:
| Metric            | Value |
| ----------------- | ----- |
| Recall (Leave)    | 0.56  |
| Precision (Leave) | 0.69  |
| F1 Score          | 0.62  |
| PR-AUC            | 0.67  |

Random Forest

Cấu hình:

```
n_estimators = 100
class_weight = balanced
```

Ngưỡng tối ưu:

```threshold = 0.30```

Kết quả: 

| Metric            | Value |
| ----------------- | ----- |
| Recall (Leave)    | 0.77  |
| Precision (Leave) | 0.40  |
| F1 Score          | 0.52  |
| PR-AUC            | 0.56  |

Random Forest đạt Recall cao hơn, phù hợp cho việc cảnh báo sớm nhân viên có nguy cơ nghỉ việc.

9. Học bán giám sát

Dự án cũng thử nghiệm self-training semi-supervised learning.

Kịch bản:

```
Chỉ 10% – 30% dữ liệu có nhãn
Phần còn lại không có nhãn
```

Quy trình:

```
Train model trên dữ liệu có nhãn
↓
Dự đoán dữ liệu chưa có nhãn
↓
Chọn các dự đoán có độ tin cậy cao
↓
Thêm pseudo-label vào tập train
↓
Huấn luyện lại mô hình
```

Tham số:

```
confidence threshold = 0.85
max iterations = 5
```

Kết quả:

| % dữ liệu có nhãn | Supervised F1 | Semi-supervised F1 |
| ----------------- | ------------- | ------------------ |
| 10%               | 0.33          | 0.21               |
| 20%               | 0.42          | 0.15               |
| 30%               | 0.39          | 0.18               |

Kết luận:

Semi-supervised không vượt qua supervised do lỗi pseudo-label.

Tuy nhiên phương pháp này có thể hữu ích khi:

```dữ liệu có nhãn < 20%```

10. Insight rút ra

Từ rule mining và clustering, các nguyên nhân chính dẫn đến nghỉ việc bao gồm:

- Lương thấp

- Nhân viên mới vào làm

- Thiếu kinh nghiệm

- Làm thêm giờ nhiều

- Ít cơ hội thăng tiến

Các insight này giúp bộ phận HR:

- Cải thiện onboarding

- Điều chỉnh chính sách lương

- Giảm áp lực làm thêm

- Xây dựng chiến lược giữ chân nhân viên

11. Hướng phát triển

Một số hướng phát triển tiếp theo:

- Deep Learning cho bài toán HR

- Explainable AI (SHAP)

- Survival Analysis để dự đoán thời điểm nghỉ việc

- Dashboard phân tích HR realtime

12. Tác giả

Trần Trường Giang

Nguyễn Nam Cường

Nguyễn Văn Đạt

Dự án bài tập lớn môn Data Mining.

13. License

Dự án này được thực hiện cho mục đích học tập và nghiên cứu.
