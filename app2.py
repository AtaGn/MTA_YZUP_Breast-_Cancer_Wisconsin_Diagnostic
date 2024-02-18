import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

class BreastCancerDiagnosisApp:
    def __init__(self):
        st.set_page_config(page_title="Milli Teknoloji Akademisi: Meme Kanseri Teşhisi Projesi", page_icon=":robot_face:", layout="centered")
        st.title('Milli Teknoloji Akademisi: Meme Kanseri Teşhisi Projesi')
        self.data = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.classifier = None
        
    def load_data(self):
        st.sidebar.header('Görev 1: Veri Seti Yükleme')
        uploaded_file = st.sidebar.file_uploader("Veri seti seçin", type=['csv'])
        if uploaded_file is not None:
            self.data = pd.read_csv(uploaded_file)
            st.write("Veri Setinin İlk 10 Satırı", self.data.head(10))
            st.write("Veri Seti Sütunları", self.data.columns.tolist())
            return True
        return False

    def preprocess_data(self):
        st.write('# Görev 2: Veri Seti Yükleme')
        self.data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
        st.write("Temizlenmiş Veri Setinin Son 10 Satırı", self.data.tail(10))
        labelencoder_Y = LabelEncoder()
        self.data['diagnosis'] = labelencoder_Y.fit_transform(self.data['diagnosis'])
        plt.figure(figsize=(10,10))
        sns.heatmap(self.data.corr(), cmap="RdYlGn")
        st.pyplot(plt)
        self.split_data()
        self.MalignantBenignPlot()
        self.scale_features()

    def MalignantBenignPlot(self):
        malignant = self.data[self.data['diagnosis'] == 1]
        benign = self.data[self.data['diagnosis'] == 0]
        fig, ax = plt.subplots()
        ax.scatter(malignant['radius_mean'], malignant['texture_mean'], color='red', label='Malignant', alpha=0.5)
        ax.scatter(benign['radius_mean'], benign['texture_mean'], color='green', label='Benign', alpha=0.5)
        ax.set_xlabel('Radius Mean')
        ax.set_ylabel('Texture Mean')
        ax.legend()
        st.pyplot(fig)

    def split_data(self):
        X = self.data.drop(['diagnosis'], axis="columns")
        Y = self.data['diagnosis']
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    def scale_features(self):
        # Öznitelikleri ölçeklendirme
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def model_selection_and_training(self):
        st.sidebar.header('Görev 3: Model Seçimi')
        classifier_name = st.sidebar.selectbox("Classifier seçin", ("KNN", "SVM", "Naive Bayes"))
        params = self.add_parameter_ui(classifier_name)
        if classifier_name in ["KNN", "SVM"]:
            self.classifier = self.perform_grid_search(self.X_train, self.Y_train, classifier_name, params)
        
        self.classifier = self.get_classifier(classifier_name, params)
        self.classifier.fit(self.X_train, self.Y_train)

    def add_parameter_ui(self, clf_name):
        params = dict()
        if clf_name == "KNN":
            K = st.sidebar.slider("K", 1, 30)
            params["n_neighbors"] = K
        elif clf_name == "SVM":
            C = st.sidebar.slider("C", 10**-2, 10.0**3)
            gamma = st.sidebar.slider("gamma", 0.000001, 1.0, step=0.000001)
            params["C"] = C
            params["gamma"] = gamma
        return params

    def get_classifier(self, clf_name, params):
        if clf_name == "KNN":
            return KNeighborsClassifier(n_neighbors=params["n_neighbors"])
        elif clf_name == "SVM":
            return SVC(C=params["C"], gamma=params["gamma"])
        else:
            return GaussianNB()

    def perform_grid_search(self, X_train, Y_train, clf_name, params):
        st.write("Gridsearch: ")
        if clf_name == "KNN":
            param_grid = {'n_neighbors': list(range(1, 31))}
            grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
            grid_search.fit(X_train, Y_train)
            best_params = grid_search.best_params_
            st.write(f"En iyi parametreler: {best_params}")
            st.write(f"Train Accuracy: {grid_search.score(X_train, Y_train)}")
            st.write(f"Test Accuracy: {grid_search.score(self.X_test, self.Y_test)}")
            return KNeighborsClassifier(**best_params)
        elif clf_name == "SVM":
            param_grid = {'C': [10**(i-2) for i in range(6)], 'gamma': [10**(-i-1) for i in range(6)]}
            grid_search = GridSearchCV(SVC(), param_grid, cv=5)
            grid_search.fit(X_train, Y_train)
            best_params = grid_search.best_params_
            st.write(f"En iyi parametreler: {best_params}")
            st.write(f"Train Accuracy: {grid_search.score(X_train, Y_train)}")
            st.write(f"Test Accuracy: {grid_search.score(self.X_test, self.Y_test)}")
            return SVC(**best_params)
        else:
            return GaussianNB()

    def model_analysis(self):
        st.write('# Görev 4: Model Analizi')
        Y_pred = self.classifier.predict(self.X_test)
        st.write(f"Model: {type(self.classifier).__name__}")
        st.write("Test Accuracy:", self.classifier.score(self.X_test, self.Y_test))
        st.write("Precision, Recall, F1-Score ve Confusion Matrix:")
        st.text("ㅤ\n"+classification_report(self.Y_test, Y_pred))
        cm = confusion_matrix(self.Y_test, Y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, ax=ax, cmap="RdYlGn")
        ax.set_xlabel('Tahmin Edilen')
        ax.set_ylabel('Gerçek')
        st.pyplot(fig)

    def run(self):
        if self.load_data():
            self.preprocess_data()
            self.model_selection_and_training()
            self.model_analysis()

app = BreastCancerDiagnosisApp()
app.run()
