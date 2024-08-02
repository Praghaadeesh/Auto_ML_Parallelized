# import pandas as pd
# import numpy as np
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import IsolationForest
# from imblearn.over_sampling import SMOTE
#
#
# class PrintImputer(SimpleImputer):
#     def fit(self, X, y=None):
#         print(f"Imputation will be applied with strategy='{self.strategy}'")
#         return super().fit(X, y)
#
#
# class PrintScaler(StandardScaler):
#     def fit(self, X, y=None):
#         print("Scaling will be applied")
#         return super().fit(X, y)
#
#
# class PrintEncoder(OneHotEncoder):
#     def fit(self, X, y=None):
#         print("One-Hot Encoding will be applied")
#         return super().fit(X, y)
#
#
# class AutoFeatureEngineering:
#     def __init__(self):
#         self.preprocessor = None
#         self.column_analysis = None
#         self.id_features = None
#         self.feature_names = None
#         self.outlier_detector = None
#
#     def read_csv(self, csv_file_path, target_column=None):
#         data = pd.read_csv(csv_file_path)
#         if target_column:
#             if target_column not in data.columns:
#                 raise ValueError(f"Target column '{target_column}' not found in CSV file.")
#             X = data.drop(target_column, axis=1)
#             y = data[target_column]
#         else:
#             X = data
#             y = None
#         return X, y
#
#     def analyze_columns(self, X):
#         analysis_results = {}
#         for column in X.columns:
#             column_data = X[column]
#             null_count = column_data.isnull().sum()
#             contains_strings = column_data.dtype == 'object'
#             distinct_count = column_data.nunique()
#             all_distinct = distinct_count == len(column_data)
#             analysis_results[column] = {
#                 'null_count': null_count,
#                 'contains_strings': contains_strings,
#                 'distinct_count': distinct_count,
#                 'all_rows_distinct': all_distinct
#             }
#         self.column_analysis = analysis_results
#
#     def fit_transform(self, csv_file_path, target_column=None):
#         X, y = self.read_csv(csv_file_path, target_column)
#         print("Original DataFrame:")
#         print(X.head())
#
#         self.analyze_columns(X)
#
#         numeric_features = []
#         categorical_features = []
#         self.id_features = []
#
#         for column, analysis in self.column_analysis.items():
#             if analysis['all_rows_distinct'] and analysis['contains_strings']:
#                 self.id_features.append(column)
#             elif analysis['contains_strings']:
#                 if analysis['distinct_count'] <= 10:
#                     categorical_features.append(column)
#                 else:
#                     self.id_features.append(column)
#             else:
#                 numeric_features.append(column)
#
#         # Define transformers
#         numeric_transformer = Pipeline(steps=[
#             ('imputer', PrintImputer(strategy='mean')),
#             ('scaler', PrintScaler())
#         ])
#
#         transformers = [('num', numeric_transformer, numeric_features)]
#
#         if categorical_features:
#             categorical_transformer = Pipeline(steps=[
#                 ('imputer', PrintImputer(strategy='constant', fill_value='missing')),
#                 ('encoder', PrintEncoder(handle_unknown='ignore', sparse_output=False))
#             ])
#             transformers.append(('cat', categorical_transformer, categorical_features))
#
#         self.preprocessor = ColumnTransformer(transformers, remainder='drop')
#
#         # Remove ID features
#         X = X.drop(columns=self.id_features)
#         print("\nDataFrame after removing ID features:")
#         print(X.head())
#
#         # Fit and transform the data
#         X_transformed = self.preprocessor.fit_transform(X, y)
#         print("\nDataFrame after preprocessing:")
#         print(pd.DataFrame(X_transformed).head())
#
#         # Get feature names after transformation
#         numeric_features_out = numeric_features
#         if categorical_features:
#             categorical_features_out = self.preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_features)
#         else:
#             categorical_features_out = []
#         self.feature_names = numeric_features_out + list(categorical_features_out)
#
#         # Convert to DataFrame
#         X_transformed = pd.DataFrame(X_transformed, columns=self.feature_names)
#         print("\nDataFrame with feature names:")
#         print(X_transformed.head())
#
#         # Outlier detection
#         self.outlier_detector = IsolationForest(contamination=0.1, random_state=42)
#         outlier_labels = self.outlier_detector.fit_predict(X_transformed)
#         X_transformed = X_transformed[outlier_labels == 1]
#         if y is not None:
#             y = y[outlier_labels == 1]
#         print("\nDataFrame after outlier removal:")
#         print(X_transformed.head())
#
#         # Balance the dataset for classification problems
#         if y is not None and len(np.unique(y)) < 10:
#             smote = SMOTE(random_state=42)
#             X_transformed, y = smote.fit_resample(X_transformed, y)
#             print("\nDataFrame after SMOTE:")
#             print(X_transformed.head())
#
#         return X_transformed, y
#
#     def transform(self, csv_file_path):
#         X, _ = self.read_csv(csv_file_path)
#         X = X.drop(columns=self.id_features)
#         print("Original DataFrame:")
#         print(X.head())
#
#         X_transformed = self.preprocessor.transform(X)
#         X_transformed = pd.DataFrame(X_transformed, columns=self.feature_names)
#         print("\nDataFrame after preprocessing:")
#         print(X_transformed.head())
#
#         outlier_labels = self.outlier_detector.predict(X_transformed)
#         X_transformed = X_transformed[outlier_labels == 1]
#         print("\nDataFrame after outlier removal:")
#         print(X_transformed.head())
#
#         return X_transformed


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")


class PrintImputer(SimpleImputer):
    def fit(self, X, y=None):
        print(f"Imputation will be applied with strategy='{self.strategy}'")
        return super().fit(X, y)


class PrintScaler(StandardScaler):
    def fit(self, X, y=None):
        print("Scaling will be applied")
        return super().fit(X, y)


class PrintEncoder(OneHotEncoder):
    def fit(self, X, y=None):
        print("One-Hot Encoding will be applied")
        return super().fit(X, y)


class AutoFeatureEngineering:
    def __init__(self):
        self.preprocessor = None
        self.column_analysis = None
        self.id_features = None
        self.feature_names = None
        self.outlier_detector = None
        self.label_encoder = None

    def read_csv(self, csv_file_path, target_column=None):
        data = pd.read_csv(csv_file_path)
        if target_column:
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in CSV file.")
            X = data.drop(target_column, axis=1)
            y = data[target_column]
        else:
            X = data
            y = None
        return X, y

    def analyze_columns(self, X):
        analysis_results = {}
        for column in X.columns:
            column_data = X[column]
            null_count = column_data.isnull().sum()
            contains_strings = column_data.dtype == 'object'
            distinct_count = column_data.nunique()
            all_distinct = distinct_count == len(column_data)
            analysis_results[column] = {
                'null_count': null_count,
                'contains_strings': contains_strings,
                'distinct_count': distinct_count,
                'all_rows_distinct': all_distinct
            }
        self.column_analysis = analysis_results

    def fit_transform(self, csv_file_path, target_column=None):
        X, y = self.read_csv(csv_file_path, target_column)
        print("Original DataFrame:")
        print(X.head())

        self.analyze_columns(X)

        numeric_features = []
        categorical_features = []
        self.id_features = []

        for column, analysis in self.column_analysis.items():
            if analysis['all_rows_distinct'] and analysis['contains_strings']:
                self.id_features.append(column)
            elif analysis['contains_strings']:
                if analysis['distinct_count'] <= 10:
                    categorical_features.append(column)
                else:
                    self.id_features.append(column)
            else:
                numeric_features.append(column)

        # Define transformers
        numeric_transformer = Pipeline(steps=[
            ('imputer', PrintImputer(strategy='mean')),
            ('scaler', PrintScaler())
        ])

        transformers = [('num', numeric_transformer, numeric_features)]

        if categorical_features:
            categorical_transformer = Pipeline(steps=[
                ('imputer', PrintImputer(strategy='constant', fill_value='missing')),
                ('encoder', PrintEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', categorical_transformer, categorical_features))

        self.preprocessor = ColumnTransformer(transformers, remainder='drop')

        # Remove ID features
        X = X.drop(columns=self.id_features)
        print("\nDataFrame after removing ID features:")
        print(X.head())

        # Fit and transform the data
        X_transformed = self.preprocessor.fit_transform(X, y)
        print("\nDataFrame after preprocessing:")
        print(pd.DataFrame(X_transformed).head())

        # Get feature names after transformation
        numeric_features_out = numeric_features
        if categorical_features:
            categorical_features_out = self.preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_features)
        else:
            categorical_features_out = []
        self.feature_names = numeric_features_out + list(categorical_features_out)

        # Convert to DataFrame
        X_transformed = pd.DataFrame(X_transformed, columns=self.feature_names)
        print("\nDataFrame with feature names:")
        print(X_transformed.head())

        # Outlier detection
        self.outlier_detector = IsolationForest(contamination=0.1, random_state=42)
        outlier_labels = self.outlier_detector.fit_predict(X_transformed)
        X_transformed = X_transformed[outlier_labels == 1]
        if y is not None:
            y = y[outlier_labels == 1]
        print("\nDataFrame after outlier removal:")
        print(X_transformed.head())

        # Balance the dataset for classification problems
        if y is not None and len(np.unique(y)) < 10:
            smote = SMOTE(random_state=42)
            X_transformed, y = smote.fit_resample(X_transformed, y)
            print("\nDataFrame after SMOTE:")
            print(X_transformed.head())

        return X_transformed, y

    def transform(self, csv_file_path):
        X, _ = self.read_csv(csv_file_path)
        X = X.drop(columns=self.id_features)
        print("Original DataFrame:")
        print(X.head())

        X_transformed = self.preprocessor.transform(X)
        X_transformed = pd.DataFrame(X_transformed, columns=self.feature_names)
        print("\nDataFrame after preprocessing:")
        print(X_transformed.head())

        outlier_labels = self.outlier_detector.predict(X_transformed)
        X_transformed = X_transformed[outlier_labels == 1]
        print("\nDataFrame after outlier removal:")
        print(X_transformed.head())

        return X_transformed

    def encode_target(self, y):
        print(f"Target datatype",y.dtype)
        if y.dtype == 'object' or y.dtype =='String':
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
            print("Target column has been label encoded")
        return y



# Initialize the feature engineering class
#feature_engineer = AutoFeatureEngineering()

# Define the path to your CSV file and the target column
#csv_file_path = f"D:/hackathon_train_data.csv" # Replace with your actual CSV file path
#target_column = 'Weighted_Price'  # Replace with your actual target column name

# Fit and transform the dataset
#X_transformed, y_transformed = feature_engineer.fit_transform(csv_file_path, target_column)

# Print the transformed features and target
#print("\nTransformed Features:")
#print(X_transformed.head(15))
#if y_transformed is not None:
#    print("\nTransformed Target:")
#    print(pd.DataFrame(feature_engineer.encode_target(y_transformed)).head(15))