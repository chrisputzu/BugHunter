# BugHunter

## Introduction
The analysis proposed within the Python script `BugHunter.py` is based on the dataset produced by the research work of the authors: Rudolf Ferenc, Péter Gyimesi, Gábor Gyimesi, Zoltán Tóth, Tibor Gyimóthy at the Department of Software Engineering and part of the MTA-SZTE Research Group on Artificial Intelligence at the University of Szeged in Hungary. Thanks to their work, encapsulated in the article 'An automatically created novel bug dataset and its validation in bug prediction' [6], published in November 2020 in the 'Journal of Systems and Software,' the authors have created a dataset useful for predicting the presence or absence of bugs in source code. Specifically, the authors selected 15 projects on Github written in the Java language and analyzed the presence of errors in their source codes at the level of:

1) Class
2) Method
3) File

For each level, the individual project's granular datasets or aggregates with all the projects are available. Furthermore, the datasets were further divided by the authors according to the method used to reduce noise in the learning set (No Filter, Removal, Subtract, Single, GDF) resulting from the temporal update of the metrics following the fixing of the bugs identified in the source code. Additionally, the dataset was projected by including both method-level and class-level metrics, along with their relationships. The authors achieved better results on the projected dataset, with the RandomForest algorithm proving to be the most performing with about 74% Precision/Recall/F-measure.

Taking this into consideration, the projected dataset 'methodp.csv' filtered with the 'Removal' method on all projects was chosen for the analysis, as it is the dataset that performs best on the validation tests conducted by the authors. Furthermore, it was chosen to execute the RandomForest algorithm on the same dataset in an attempt to replicate the results obtained by the authors.

Selected Projects (Project column):
1) Android Universal Image Loader: An Android library that assists with image loading.
2) ANTLR v4: Popular software in the field of language processing. It is a powerful parser generator for reading, processing, executing, or translating structured texts or binary files.
3) Broadleaf Commerce: A framework for building e-commerce websites.
4) Elasticsearch: A very popular RESTful search engine.
5) Eclipse plugin for Ceylon: An Eclipse plugin that provides an Integrated Development Environment (IDE) for the Ceylon programming language.
6) Hazelcast: A platform for distributed data processing.
7) jUnit: A Java framework for writing unit tests.
8) MapDB: A versatile, fast, and easy-to-use database engine in Java.
9) mcMMO: A Minecraft-based role-playing game (RPG).
10) Mission Control Technologies: Initially developed by NASA for space flight operations. It is a real-time monitoring and visualization platform that can also be used for monitoring other data.
11) Neo4j: The world's leading graph database with high performance.
12) Netty: An asynchronous, event-driven network framework.
13) OrientDB: A popular NoSQL, document-based graph database. Mainly famous for its speed and scalability.
14) Oryx 2: An open-source software with machine learning algorithms that enables the processing of huge datasets.
15) Titan: A highly scalable, high-performance graph database.
  

## Description
BugHunter is a Python class for analyzing and modeling bug data using PySpark. The class provides methods for establishing a connection to Spark, reading data, exploring, cleaning, transforming, scaling, splitting, training, evaluating, and visualizing a random forest model for bug prediction. The BugHunter class can be used to analyze and model bug data efficiently and accurately.

## Key Features
- Establishes a connection to Spark.
- Reads bug data from a CSV file and returns a Spark DataFrame.
- Explores the loaded bug data, printing its schema, summary statistics, and a sample.
- Cleans the loaded bug data by removing duplicates and columns with all zero values.
- Transforms the cleaned bug data by processing text columns and assembling features.
- Scales the assembled bug data using MinMaxScaler.
- Splits the scaled bug data into training and testing sets.
- Trains a random forest model using the training data and returns the best model.
- Evaluates the best model's performance on the test data.
- Plots the results of the best model, including the confusion matrix, ROC curve, and important features.

## Dependencies
- PySpark: Python library for Apache Spark
- pandas: Python library for data manipulation and analysis
- matplotlib: Python library for data visualization
- scikit-learn: Python library for machine learning

## Usage
1. Clone the repository to your local machine.
2. Ensure you have installed the necessary libraries: PySpark, pandas, matplotlib, and scikit-learn.
3. Run the provided Python script for bug analysis and modeling: `BugHunter.py`.
4. Input the path to the bug data CSV file.
5. Follow the instructions in the script to explore, clean, transform, scale, split, train, evaluate, and visualize the bug data.

## Contributions
Contributions to this project are welcome! Feel free to submit pull requests for bug fixes, enhancements, or additional features. Please adhere to the project's coding style and guidelines.

## License
This project is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/).

## Disclaimer
While efforts have been made to ensure accurate bug analysis and modeling, no guarantees are provided regarding the accuracy or reliability of the system in all scenarios. Usage of this project is at the user's discretion.

## References
- PySpark Documentation: [https://spark.apache.org/docs/latest/api/python/index.html](https://spark.apache.org/docs/latest/api/python/index.html)
- Pandas Documentation: [https://pandas.pydata.org/pandas-docs/stable/index.html](https://pandas.pydata.org/pandas-docs/stable/index.html)
- Matplotlib Documentation: [https://matplotlib.org/stable/contents.html](https://matplotlib.org/stable/contents.html)
- Scikit-learn Documentation: [https://scikit-learn.org/stable/user_guide.html](https://scikit-learn.org/stable/user_guide.html)
- Scientific Paper: [An automatically created novel bug dataset and its validation in bug prediction](https://www.sciencedirect.com/science/article/pii/S0164121220301436?via%3Dihub#tbl8)

## Acknowledgments
The authors of the scientific paper used as a reference:
- Rudolf Ferenc
- Péter Gyimesi
- Gábor Gyimesi
- Zoltán Tóth
- Tibor Gyimóthy

## Author
- Christian Putzu
