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

st.set_page_config(page_title="Milli Teknoloji Akademisi: Meme Kanseri Teşhisi Projesi", page_icon=":robot_face:", layout="centered")

# Streamlit sayfasını başlat
st.title('Milli Teknoloji Akademisi: Meme Kanseri Teşhisi Projesi', )

# Görev 1: Veri yükleme ve gösterme
st.sidebar.header('Görev 1: Veri Seti Yükleme')
uploaded_file = st.sidebar.file_uploader("Veri seti seçin", type=['csv'])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Veri Setinin İlk 10 Satırı", data.head(10))
    st.write("Veri Seti Sütunları", data.columns.tolist())

    # Görev 2: Veri ön işleme
    st.write('# Görev 2: Veri Seti Yükleme')
    data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)  # gereksiz sütunları temizle
    st.write("Temizlenmiş Veri Setinin Son 10 Satırı", data.tail(10))
    
    # 'diagnosis' sütununu 0 ve 1'e dönüştür
    labelencoder_Y = LabelEncoder()
    data['diagnosis'] = labelencoder_Y.fit_transform(data['diagnosis'])
    
    # Korelasyon matrisi
    drop_list1 = ['perimeter_mean','radius_mean','compactness_mean','concave points_mean',
                'radius_se','perimeter_se','radius_worst','perimeter_worst','compactness_worst',
                'concave points_worst','compactness_se','concave points_se','texture_worst','area_worst']
    plt.figure(figsize=(10,10))
    sns.heatmap(data.corr())
    st.pyplot(plt)
    




    # Malignant ve Benign dataları ayırma ve görselleştirme
    malignant = data[data['diagnosis'] == 1]
    benign = data[data['diagnosis'] == 0]
    fig, ax = plt.subplots()
    ax.scatter(malignant['radius_mean'], malignant['texture_mean'], color='red', label='Malignant', alpha=0.5)
    ax.scatter(benign['radius_mean'], benign['texture_mean'], color='green', label='Benign', alpha=0.5)
    ax.set_xlabel('Radius Mean')
    ax.set_ylabel('Texture Mean')
    ax.legend()
    st.pyplot(fig)
    



    # Veriyi eğitim ve test setlerine ayırma

    # X = data[['texture_mean', 'area_mean', 'smoothness_mean', 'concavity_mean',
    # 'area_se', 'concavity_se', 'fractal_dimension_se', 'smoothness_worst',
    # 'concavity_worst', 'symmetry_worst']]

    X = data.drop(['diagnosis'], axis="columns")

    Y = data['diagnosis']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


    # Görev 3: Model seçimi ve eğitimi
    st.sidebar.header('Görev 3: Model Seçimi')
    classifier_name = st.sidebar.selectbox("Classifier seçin", ("KNN", "SVM", "Naive Bayes"))
    
    def add_parameter_ui(clf_name):
        params = dict()
        if clf_name == "KNN":
            K = st.sidebar.slider("K", 1, 30)
            params["n_neighbors"] = K
        elif clf_name == "SVM":
            C = st.sidebar.slider("C", 10**-2, 10.0**3)
            gamma = st.sidebar.slider("gamma", 0.000001, 1.0)
            params["C"] = C
            params["gamma"] = gamma
        return params
    
    params = add_parameter_ui(classifier_name)
    
    def get_classifier(clf_name, params):
        clf = None
        if clf_name == "KNN":
            clf = KNeighborsClassifier(n_neighbors=params["n_neighbors"])
        elif clf_name == "SVM":
            clf = SVC(C=params["C"], gamma=params["gamma"])
        else:
            clf = GaussianNB()
        return clf
    
    

    def perform_grid_search(X_train, Y_train, clf_name, params):
        st.write("Gridsearch: ")
        if clf_name == "KNN":
            param_grid = {'n_neighbors': list(range(1, 31))}
            grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
            grid_search.fit(X_train, Y_train)
            best_params = grid_search.best_params_
            st.write(f"En iyi parametreler: {best_params}")
            st.write(f"Train Accuracy: {grid_search.score(X_train, Y_train)}")
            st.write(f"Test Accuracy: {grid_search.score(X_test, Y_test)}")
            return KNeighborsClassifier(**best_params)
        elif clf_name == "SVM":
            param_grid = {'C': [10**(i-2) for i in range(6)], 'gamma': [10**(-i-1) for i in range(6)]}
            grid_search = GridSearchCV(SVC(), param_grid, cv=5)
            grid_search.fit(X_train, Y_train)
            best_params = grid_search.best_params_
            st.write(f"En iyi parametreler: {best_params}")
            st.write(f"Train Accuracy: {grid_search.score(X_train, Y_train)}")
            st.write(f"Test Accuracy: {grid_search.score(X_test, Y_test)}")
            return SVC(**best_params)
        else:
            # Naive Bayes için grid search uygulanmaz
            return GaussianNB()
        

    # Model seçimi ve eğitimi kısmında get_classifier fonksiyonu çağrılırken

    # GridSearchCV ile en iyi parametreleri bul ve modeli bu parametrelerle eğit
    if classifier_name in ["KNN", "SVM"]:  # Naive Bayes için gerekli değil
        clf = perform_grid_search(X_train, Y_train, classifier_name, params)

    clf = get_classifier(classifier_name, params)


    clf.fit(X_train, Y_train)  # En iyi parametrelerle modeli eğit

    Y_pred = clf.predict(X_test)
    
    # Görev 4: Model analizi
    st.write('# Görev 4: Model Analizi')

    st.write(f"Model: {classifier_name}")
    st.write("Test Accuracy:", clf.score(X_test, Y_test))
    st.write("Precision, Recall, F1-Score ve Confusion Matrix:")
    st.text("ㅤ\n"+classification_report(Y_test, Y_pred))
    cm = confusion_matrix(Y_test, Y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, ax=ax)
    ax.set_xlabel('Tahmin Edilen')
    ax.set_ylabel('Gerçek')
    st.pyplot(fig)
