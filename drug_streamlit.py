import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Tạo SparkSession
spark = SparkSession.builder \
    .master("spark://10.7.143.12:7077") \
    .config('spark.driver.cores', '6') \
    .config('spark.executor.memory', '10g') \
    .config('spark.executor.cores', '8') \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Đọc dữ liệu
df = spark.read.csv("drug_data_5million.csv", header=True, inferSchema=True)

# Mã hóa các cột phân loại (Sex, BP, Cholesterol) thành số
indexer_sex = StringIndexer(inputCol="Sex", outputCol="Sex_index")
indexer_bp = StringIndexer(inputCol="BP", outputCol="BP_index")
indexer_cholesterol = StringIndexer(inputCol="Cholesterol", outputCol="Cholesterol_index")

# Gộp các đặc trưng thành một cột vector với VectorAssembler
assembler = VectorAssembler(
    inputCols=["Age", "Sex_index", "BP_index", "Cholesterol_index", "Na_to_K"],
    outputCol="features"
)

# Mã hóa nhãn (Drug) thành số với StringIndexer
label_indexer = StringIndexer(inputCol="Drug", outputCol="label")

# Khởi tạo mô hình Naive Bayes của PySpark
nb = NaiveBayes(featuresCol="features", labelCol="label")

# Tạo pipeline bao gồm các bước tiền xử lý và mô hình
pipeline = Pipeline(stages=[indexer_sex, indexer_bp, indexer_cholesterol, assembler, label_indexer, nb])

# Chia dữ liệu thành tập huấn luyện và kiểm tra
train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)

# Huấn luyện mô hình trên tập huấn luyện
model = pipeline.fit(train_df)

# Dự đoán trên tập kiểm tra
predictions = model.transform(test_df)

# Đánh giá độ chính xác
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)

# Tạo giao diện Streamlit
st.title("Ứng dụng Phân loại với Naive Bayes")
st.write(f"Độ chính xác của mô hình: {accuracy * 100:.2f}%")

# Form nhập liệu cho người dùng
with st.form(key='prediction_form'):
    st.subheader("Nhập thông tin bệnh nhân")
    
    age_input = st.number_input('Nhập tuổi:', min_value=18, max_value=100, value=30)
    sex_input = st.selectbox('Chọn giới tính:', ['M', 'F'])
    bp_input = st.selectbox('Chọn huyết áp:', ['HIGH', 'LOW', 'NORMAL'])
    cholesterol_input = st.selectbox('Chọn mức cholesterol:', ['HIGH', 'NORMAL'])
    na_to_k_input = st.number_input('Nhập tỷ lệ Na/K:', min_value=10.0, max_value=25.0, value=15.0)
    
    # Nút dự đoán
    predict_button = st.form_submit_button(label='Dự đoán loại thuốc')

# Hiển thị kết quả khi nút dự đoán được nhấn
if 'predict_button_clicked' not in st.session_state:
    st.session_state.predict_button_clicked = False

if predict_button or st.session_state.predict_button_clicked:
    st.session_state.predict_button_clicked = True
    
    # Tạo Spark DataFrame từ dữ liệu đầu vào của người dùng
    input_row = Row(Age=age_input, Sex=sex_input, BP=bp_input, Cholesterol=cholesterol_input, **{'Na_to_K': na_to_k_input})
    input_df = spark.createDataFrame([input_row])
    
    # Hiển thị thông báo đang dự đoán
    with st.spinner('Đang thực hiện dự đoán...'):
        # Áp dụng cùng pipeline để mã hóa đầu vào của người dùng
        prediction_df = model.transform(input_df)
        predicted_index = prediction_df.select("prediction").collect()[0][0]
        
        # Ensure the label mapping is retrieved from the StringIndexerModel
        label_indexer_model = model.stages[4]  # Assuming the 5th stage is the StringIndexer for labels
        predicted_drug = label_indexer_model.labels[int(predicted_index)]  # Lấy nhãn dự đoán từ StringIndexer
    
    # Hiển thị kết quả với hiệu ứng thành công
    st.success(f"Loại thuốc được đề xuất: {predicted_drug}")
    
    # Hiển thị thông tin chi tiết
    st.subheader("Thông tin bệnh nhân")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Tuổi:** {age_input}")
        st.write(f"**Giới tính:** {sex_input}")
        st.write(f"**Huyết áp:** {bp_input}")
    with col2:
        st.write(f"**Cholesterol:** {cholesterol_input}")
        st.write(f"**Tỷ lệ Na/K:** {na_to_k_input}")