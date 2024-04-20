import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import DataFrame
from typing import Tuple
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, desc
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, RegexTokenizer, StopWordsRemover, HashingTF, IDF, Binarizer, MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc



class BugHunter():
    """
    A class for analyzing and modeling bug data.

    Parameters:
    -----------
    MASTER_IP : str
        The IP address of the master node for Spark connection.

    Attributes:
    -----------
    MASTER_IP : str
        The IP address of the master node for Spark connection.

    Methods:
    --------
    create_connection(self):
        Establishes a connection to Spark.

    read_data(self, data: str, spark: SparkSession) -> DataFrame:
        Reads the data and returns a Spark DataFrame.

    explore_data(self, loaded_data: DataFrame):
        Explores the loaded data, printing its schema, summary statistics, and a sample.

    clean_data(self, loaded_data: DataFrame) -> DataFrame:
        Cleans the loaded data by removing duplicates and columns with all zero values.

    transform_data(self, cleaned_data: DataFrame) -> DataFrame:
        Transforms the cleaned data by processing text columns and assembling features.

    scale_data(self, assembled_data: DataFrame) -> DataFrame:
        Scales the assembled data using MinMaxScaler.

    split_data(self, scaled_data: DataFrame) -> Tuple[DataFrame, DataFrame]:
        Splits the scaled data into training and testing sets.

    fit_model(self, train_data: DataFrame) -> RandomForestClassificationModel:
        Trains a random forest model using the training data and returns the best model.

    evaluate_model(self, best_model: RandomForestClassificationModel, test_data: DataFrame) -> DataFrame:
        Evaluates the best model's performance on the test data.

    plot_results_model(self, predictions: DataFrame):
        Plots the results of the best model, including the confusion matrix, ROC curve, and important features.
    """
    def __init__(self, MASTER_IP):
        
        self.MASTER_IP = MASTER_IP
        
    def create_connection(self):
        """
        Establishes a connection to Spark.

        Returns:
        --------
        spark : SparkSession
            The SparkSession object.
        """
        conf = SparkConf()\
            .setAppName('BugHunter')\
            .setMaster(f"spark://{MASTER_IP}:7077")\
            .set("spark.driver.host", f"{MASTER_IP}")\
            .set("spark.driver.port", "7077")\
            .set("spark.driver.bindAddress", "0.0.0.0")\
            .set("spark.executor.memory", "8g")\
            .set("spark.driver.memory", "8g")\
            .set("spark.task.maxFailures", "300")\
            .set("spark.shuffle.io.maxRetries", "100")\
            .set("spark.network.timeout", "60000000s")\
            .set("spark.executor.heartbeatInterval", "6s")\
            .set("spark.sql.shuffle.partitions", "8")\
            .set("spark.shuffle.compress", "true")\
            .set("spark.reducer.maxSizeInFlight", "256m")
        
        spark = SparkSession.builder\
            .config(conf=conf)\
            .getOrCreate()
        
        spark.sparkContext.setLogLevel("ERROR")
        
        return spark
    
    def read_data(self, data: str, spark: SparkSession) -> DataFrame:
        """
        Reads the data and returns a Spark DataFrame.

        Parameters:
        -----------
        data : str
            The path to the data file.
        spark : SparkSession
            The SparkSession object.

        Returns:
        --------
        loaded_data : pyspark.sql.DataFrame
            The loaded data as a Spark DataFrame.
        """
        print("\n 1) Step - Loading data: ", data)
        
        df = pd.read_csv(data)
        loaded_data = spark.createDataFrame(df)
        
        return loaded_data

        
    def explore_data(self, loaded_data: DataFrame):
        """
        Explores the loaded data, printing its schema, summary statistics, and a sample.

        Parameters:
        -----------
        loaded_data : pyspark.sql.DataFrame
            The loaded data as a Spark DataFrame.
        """
        print("\n 2) Step - Exploring data: ")
    
        loaded_data.printSchema() 
        print(loaded_data.describe().toPandas()) 
        print(loaded_data.limit(15).toPandas())
        num_rows = loaded_data.count()
        print("Number of rows:", num_rows) 
        num_columns = len(loaded_data.columns)
        print("Number of columns:", num_columns) 
        
        nan_counts = loaded_data.count() - loaded_data.dropna().count()        
        print("Number of Na values:", nan_counts)  
        empty_count = sum(loaded_data.where(col(column) == "").count()
                            for column in loaded_data.columns)
        
        print("Number of empty values:", empty_count) 
        null_count = sum(loaded_data.filter(col(column).isNull()).count() for column in loaded_data.columns)
        print("Number of NULL values:", null_count) 
        duplicates_counts = loaded_data.count() - \
            loaded_data.dropDuplicates().count()
        
        print("Number of duplicates:", duplicates_counts) 
        
        print('Number of bugs by Project: ')
        project_bugs = loaded_data.groupBy("Project") \
                                    .agg(F.sum("Number of Bugs").alias("Total_Bugs")) \
                                    .orderBy(F.desc("Total_Bugs"))
        project_bugs.show(truncate=False)
        hash_bug_totals = loaded_data.groupBy("Hash") \
                                .agg(F.sum("Number of Bugs").alias("Total_Bugs")) \
                                .orderBy(F.desc("Total_Bugs"))
        print("Bug totals grouped by Hash:")
        hash_bug_totals.show(truncate=False)
        longname_bug_totals = loaded_data.groupBy("LongName") \
                                        .agg(F.sum("Number of Bugs").alias("Total_Bugs")) \
                                        .orderBy(F.desc("Total_Bugs"))
        print("Bug totals grouped by LongName:")
        longname_bug_totals.show(truncate=False)
        parent_bug_totals = loaded_data.groupBy("Parent") \
                                    .agg(F.sum("Number of Bugs").alias("Total_Bugs")) \
                                    .orderBy(F.desc("Total_Bugs"))
        print("Bug totals grouped by Parent:")
        parent_bug_totals.show(truncate=False)
        total_bugs = loaded_data.select(F.sum("Number of Bugs")).collect()[0][0]
        print("Total number of bugs:", total_bugs)
        
        unique_projects_count = loaded_data.select("Project").distinct().count()
        print("Number of unique values for Projects column:", unique_projects_count)
        print("Number of unique values for Hash column:", loaded_data.select("Hash").distinct().count())
        print("Number of unique values for LongName column:", loaded_data.select("LongName").distinct().count())
        print("Number of unique values for Parent column:", loaded_data.select("Parent").distinct().count())
        print("Unique values and count for Hash column:")
        loaded_data.groupBy("Hash").count().orderBy("count", ascending=False).show(truncate=False)
        print("Unique values and count for LongName column:")
        loaded_data.groupBy("LongName").count().orderBy("count", ascending=False).show(truncate=False)
        print("Unique values and count for Parent column:")
        loaded_data.groupBy("Parent").count().orderBy("count", ascending=False).show(truncate=False)
        
        for column in loaded_data.columns:
            print("Unique values and count for", column, "column (ordered by count in descending order):")
            loaded_data.groupBy(column).count().orderBy(desc("count")).show()
            total_rows = loaded_data.count()
            zero_count = loaded_data.where(col(column) == 0).count()
            zero_percentage = (zero_count / total_rows) * 100
            print(f"Percentage of zeros in {column} column: {zero_percentage:.2f}%")
        

    def clean_data(self, loaded_data: DataFrame) -> DataFrame:
        """
        Cleans the loaded data by removing duplicates and columns with all zero values.

        Parameters:
        -----------
        loaded_data : pyspark.sql.DataFrame
            The loaded data as a Spark DataFrame.

        Returns:
        --------
        cleaned_data : pyspark.sql.DataFrame
            The cleaned data as a Spark DataFrame.
        """
        print("\n 3) Step - Data Cleaning: ")
        
        print("\nNumber of rows before removing duplicates:", loaded_data.count()) 
        cleaned_data = loaded_data.dropDuplicates()
        num_duplicates_removed = loaded_data.count() - cleaned_data.count()
        print(f"Number of duplicates removed: {num_duplicates_removed}") 
        print("Number of rows after removing duplicates:", cleaned_data.count()) 
        
        num_cols_before = len(loaded_data.columns)
        print(f"Number of columns before the drop: {num_cols_before}")
        print("\nColumns to drop 100% of zero values: ")
        columns_to_drop = ["Hash",
        "WarningBlocker", "WarningInfo", "Android Rules",
         "Code Size Rules", "Finalizer Rules",
        "Comment Rules", "Coupling Rules", 
        "JavaBean Rules", "MigratingToJUnit4 Rules",
        "Migration13 Rules", "Migration14 Rules",
        "Migration15 Rules", "Vulnerability Rules"
        ]
        cleaned_data = cleaned_data.drop(*columns_to_drop)
        num_cols_after = len(cleaned_data.columns)
        print(f"Number of columns removed: {num_cols_before - num_cols_after}")
        print(f"Number of columns after the drop: {num_cols_after}")
        
        # Downsampling target variable
        num_zero_bugs = cleaned_data.filter(col("Number of Bugs") == 0).count()
        num_non_zero_bugs = cleaned_data.filter(col("Number of Bugs") != 0).count()
        min_rows_per_class = min(num_zero_bugs, num_non_zero_bugs)
        
        zero_bugs_sampled = cleaned_data.filter(col("Number of Bugs") == 0) \
                .sample(False, min_rows_per_class / num_zero_bugs)
        non_zero_bugs_sampled = cleaned_data.filter(col("Number of Bugs") != 0) \
                .sample(False, min_rows_per_class / num_non_zero_bugs)
        cleaned_data = zero_bugs_sampled.union(non_zero_bugs_sampled)
        
        print("\nDebug of downsampling: ")
        print(f"Balanced by min 'Number of Bugs' non-zero class: {min_rows_per_class}")
        print(f"Number of zero values on Number of Bugs column: {zero_bugs_sampled.count()}")
        print(f"Number of non-zero values on Number of Bugs column: {non_zero_bugs_sampled.count()}")

        return cleaned_data

        
    def transform_data(self, cleaned_data: DataFrame) -> DataFrame:
        """
        Transforms the cleaned data by processing text columns and assembling features.

        Parameters:
        -----------
        cleaned_data : pyspark.sql.DataFrame
            The cleaned data as a Spark DataFrame.

        Returns:
        --------
        assembled_data : pyspark.sql.DataFrame
            The transformed data with assembled features as a Spark DataFrame.
        """
        print("\n 4) Step - Data Transformation: ")
    
        numeric_data = cleaned_data.columns[3:62]
        print(numeric_data)
        cleaned_data = cleaned_data.withColumn("Number of Bugs", 
                                            col("Number of Bugs").cast("double"))
        project_indexer = StringIndexer(inputCol="Project", 
                                        outputCol="ProjectIndex")
        project_encoder = OneHotEncoder(inputCol="ProjectIndex", 
                                        outputCol="ProjectVec")
        project_pipeline = Pipeline(stages=[project_indexer, project_encoder])
        project_encoded = project_pipeline.fit(cleaned_data).transform(cleaned_data)
        
        text_columns = ["LongName", "Parent"]
        text_pipeline_stages = []
        for column in text_columns:
            tokenizer = RegexTokenizer(pattern="[./();$<>]",
                                    inputCol=column, 
                                    outputCol=column +'_tokenized')
            stopwords_remover = StopWordsRemover(inputCol=column + '_tokenized',
                                                outputCol=column +'_no-stopwords', 
                                                stopWords=['v', 'l', 'z', 's', 'i', 'd', 'f', 'e', 'j'])
            hashingTF = HashingTF(inputCol=column + "_no-stopwords", 
                                outputCol=column + "_tf", 
                                numFeatures=1000)
            idf = IDF(inputCol=column + "_tf", 
                    outputCol=column + "_tfidf")
            text_pipeline_stages.extend([tokenizer, stopwords_remover, hashingTF, idf])
        
        text_pipeline = Pipeline(stages=text_pipeline_stages)
        text_data = text_pipeline.fit(project_encoded).transform(project_encoded)
        text_data.select(text_data.columns[-11:]).show()

            
        binarizer = Binarizer(threshold=0.0, 
                            inputCol="Number of Bugs", 
                            outputCol="Number of Bugs Binary")
        
        assembler = VectorAssembler(inputCols=["ProjectVec"] + 
                                    ["LongName_tfidf"] + 
                                    ["Parent_tfidf"] + 
                                    numeric_data,
                                    outputCol="features")
        
        final_pipeline = Pipeline(stages=[binarizer, assembler])
        assembled_data = final_pipeline.fit(text_data).transform(text_data)
        
        print('Debug transformed_data: ')
        assembled_data.printSchema()
        assembled_data.select(assembled_data.columns[-13:]).show()
        
        return assembled_data
    
       
    def scale_data(self, assembled_data: DataFrame) -> DataFrame:
        """
        Scales the assembled data using MinMaxScaler.

        Parameters:
        -----------
        assembled_data : pyspark.sql.DataFrame
            The assembled data with features as a Spark DataFrame.

        Returns:
        --------
        scaled_data : pyspark.sql.DataFrame
            The scaled data as a Spark DataFrame.
        """
        print("\n 5) Step - Data Scaling: ")

        scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
        scaler_model = scaler.fit(assembled_data)
        scaled_data = scaler_model.transform(assembled_data)
        scaled_data.select('features').show(truncate=False)
        scaled_data.select('scaledFeatures').show(truncate=False)
            
        return scaled_data
    
    def split_data(self, scaled_data: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """
        Splits the scaled data into training and testing sets.

        Parameters:
        -----------
        scaled_data : pyspark.sql.DataFrame
            The scaled data as a Spark DataFrame.

        Returns:
        --------
        train_data : pyspark.sql.DataFrame
            The training data as a Spark DataFrame.
        test_data : pyspark.sql.DataFrame
            The testing data as a Spark DataFrame.
        """
        print("\n 6) Step - Data Splitting: ")

        selected = scaled_data.select('scaledFeatures','Number of Bugs Binary')
        selected.show()
        
        train_data, test_data = selected.randomSplit([0.8, 0.2], seed=13)
        print("Number of rows into Training set: ", train_data.count())
        train_data.show()
        print("Number of rows into Test set: ", test_data.count())
        test_data.show()
        
        return train_data, test_data
    
    def fit_model(self, train_data: DataFrame) -> DataFrame:
        """
        Trains a random forest model using the training data and returns the best model.

        Parameters:
        -----------
        train_data : pyspark.sql.DataFrame
            The training data as a Spark DataFrame.

        Returns:
        --------
        best_model : RandomForestClassificationModel
            The trained best model.
        """
        print("\n 7) Step - Model Training: ")

        random_forest = RandomForestClassifier(labelCol="Number of Bugs Binary", 
                                               featuresCol='scaledFeatures', 
                                               impurity='entropy')
        best_model = random_forest.fit(train_data)
        
        param_grid = ParamGridBuilder() \
                .addGrid(random_forest.maxDepth, [5, 10, 15, 20]) \
                .addGrid(random_forest.numTrees, [5, 50, 100, 150]) \
                .build()
                
        evaluator = BinaryClassificationEvaluator(labelCol="Number of Bugs Binary", 
                                                  metricName="areaUnderROC")

        crossval = CrossValidator(estimator=random_forest,
                                estimatorParamMaps=param_grid,
                                evaluator=evaluator,
                                numFolds=5,
                                parallelism=8)
        
        cv_model = crossval.fit(train_data)
        avg_metrics = cv_model.avgMetrics
        print("ROC AUC results from cv on training set:", avg_metrics)

        best_model = cv_model.bestModel

        print("Best model parameters:")
        print("maxDepth:", best_model._java_obj.getMaxDepth())
        print("numTrees:", best_model._java_obj.getNumTrees())
        
        return best_model
            
    def evaluate_model(self, best_model: RandomForestClassificationModel, test_data: DataFrame) -> DataFrame:
        """
        Evaluates the best model's performance on the test data.

        Parameters:
        -----------
        best_model : RandomForestClassificationModel
            The trained best model.
        test_data : pyspark.sql.DataFrame
            The testing data as a Spark DataFrame.

        Returns:
        --------
        predictions : pyspark.sql.DataFrame
            The predictions made by the best model on the test data.
        """
        print("\n 8) Step - Model Evaluation: ")

        predictions = best_model.transform(test_data)
        predictions.show(10)
        
        evaluator_f1 = MulticlassClassificationEvaluator(labelCol="Number of Bugs Binary", 
                                                         metricName="f1")
        f1_score = evaluator_f1.evaluate(predictions)
        print(f"F1 Score:{f1_score:.4f}") 
        
        evaluator_acc = MulticlassClassificationEvaluator(labelCol="Number of Bugs Binary", 
                                                          metricName="accuracy")
        accuracy = evaluator_acc.evaluate(predictions)
        print(f"Accuracy: {accuracy:.4f}") 
        
        evaluator_recall = MulticlassClassificationEvaluator(labelCol="Number of Bugs Binary", 
                                                             metricName="weightedRecall")
        recall = evaluator_recall.evaluate(predictions)
        print(f"Recall: {recall:.4f}")
        
        evaluator_precision = MulticlassClassificationEvaluator(labelCol="Number of Bugs Binary", 
                                                                metricName="weightedPrecision")
        precision = evaluator_precision.evaluate(predictions)
        print(f"Precision: {precision:.4f}") 
        
        evaluator_roc = BinaryClassificationEvaluator(labelCol="Number of Bugs Binary", 
                                                            metricName="areaUnderROC")
        roc = evaluator_roc.evaluate(predictions)
        print(f"ROC AUC: {roc:.4f}")
            
        return predictions
            
                        
    def plot_results_model(self, predictions: DataFrame):
        """
        Plots the results of the best model, including the confusion matrix, ROC curve, and important features.

        Parameters:
        -----------
        predictions : pyspark.sql.DataFrame
            The predictions made by the best model on the test data.
        """
        print("\n 9) Step - Model Performance Visualization: ")

        y_true = predictions.select("Number of Bugs Binary").rdd.flatMap(lambda x: x).collect()
        y_pred = predictions.select("prediction").rdd.flatMap(lambda x: x).collect()
        
        conf_matrix = confusion_matrix(y_true, y_pred)
        display_matrix = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, 
                                                display_labels=[0, 1])
        display_matrix.plot()
        plt.title("Confusion Matrix")
        plt.show()
        
        true_neg, false_pos, false_neg, true_pos = conf_matrix.ravel()
        print("True Negative:", true_neg)
        print("False Positive:", false_pos)
        print("False Negative:", false_neg)
        print("True Positive:", true_pos)
        
        y_prob = predictions.select("probability").rdd.map(lambda x: x[0][1]).collect()
        false_pred, true_pred, _ = roc_curve(y_true, y_prob)
        roc_area = auc(false_pred, true_pred)
        plt.plot(false_pred, true_pred, color="green", lw=2, 
                label=f"ROC Curve AUC: {roc_area:.4f})")
        plt.plot([0, 1], [0, 1], color="purple", lw=2, linestyle="-")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic Curve")
        plt.legend(loc="lower right")
        plt.show()
        
        importance_features = best_model.featureImportances.toArray()
        sorted_list_indices = importance_features.argsort()[::-1]
        top_features_indices = sorted_list_indices[:15]
        top_features_importance = importance_features[top_features_indices]

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(top_features_importance)), top_features_importance, color = 'green')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.title('Top 15 Important Features')
        plt.xticks(range(len(top_features_importance)), top_features_indices)
        plt.show()  
        
        print("\nTop 15 Important Features:")
        for i in range(15):
            feature_index = sorted_list_indices[i]
            feature_importance = importance_features[feature_index]
            print(f"Feature {feature_index}: {feature_importance:.4f}")
    
    
if __name__=="__main__":
    
    data = "method-p.csv"
    
    MASTER_IP = input("Insert your Master IP: ")
            
    bh = BugHunter(MASTER_IP=MASTER_IP)
    spark = bh.create_connection()
    loaded_data = bh.read_data(data=data, spark=spark)
    bh.explore_data(loaded_data)
    cleaned_data = bh.clean_data(loaded_data=loaded_data)
    assembled_data = bh.transform_data(cleaned_data=cleaned_data)
    scaled_data = bh.scale_data(assembled_data=assembled_data)
    train_data, test_data = bh.split_data(scaled_data=scaled_data)
    best_model = bh.fit_model(train_data=train_data)
    predictions = bh.evaluate_model(best_model=best_model, test_data=test_data)
    bh.plot_results_model(predictions=predictions)
    spark.stop()
