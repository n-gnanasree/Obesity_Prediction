import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

class ObesityModel:
    def __init__(self, csv_path="Obesity prediction.csv"):
        self.df = pd.read_csv(csv_path)
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.target_encoder = LabelEncoder()
        self.model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        self._prepare_data()

    def _prepare_data(self):
        self.df.rename(columns=lambda x: x.strip(), inplace=True)
        if 'Obesity' not in self.df.columns:
            raise ValueError("Target column 'Obesity' not found in dataset.")

        cat_cols = self.df.select_dtypes(include='object').columns.tolist()
        if 'Obesity' in cat_cols:
            cat_cols.remove('Obesity')

        for col in cat_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le

        self.df["Obesity"] = self.target_encoder.fit_transform(self.df["Obesity"])
        self.X = self.df.drop(columns=["Obesity"])
        self.y = self.df["Obesity"]

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

        self.model.fit(self.X_train_scaled, self.y_train)

    def predict(self, input_data):
        return self.model.predict(self.scaler.transform([input_data]))[0]

    def get_metrics(self):
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        y_pred = self.model.predict(self.X_test_scaled)
        return {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "precision": precision_score(self.y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
        }

    def get_feature_names(self):
        return self.X.columns.tolist()

    def get_label_encoders(self):
        return self.label_encoders

    def get_target_encoder(self):
        return self.target_encoder
