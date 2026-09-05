from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import MLConfigForm

import io
import base64
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def _generate_dummy_dataset(n_rows: int = 36) -> pd.DataFrame:
    kondisi_options = ['Cerah', 'Berawan', 'Mendung']
    rows = []
    for i in range(n_rows):
        suhu = 22 + (i % 7)
        kelembapan = 55 + ((i * 3) % 40)
        kondisi = kondisi_options[i % 3]
        hujan_besok = 'Ya' if (kelembapan >= 75 or kondisi == 'Mendung') else 'Tidak'
        rows.append({
            'Suhu': suhu,
            'Kelembapan': kelembapan,
            'KondisiAwan': kondisi,
            'HujanBesok': hujan_besok,
        })
    df = pd.DataFrame(rows)
    # Sisipkan NaN untuk demonstrasi imputasi
    df.loc[5, 'Suhu'] = float('nan')
    df.loc[15, 'Kelembapan'] = float('nan')
    return df


def _preprocess(df: pd.DataFrame, use_ohe: bool = True, impute_median: bool = True):
    # One-Hot Encoding fitur kategorikal jika diminta
    if use_ohe:
        df = pd.get_dummies(df, columns=['KondisiAwan'], prefix='Awan', drop_first=False)

    # Label encoding target
    label_map = {'Ya': 1, 'Tidak': 0}
    df['HujanBesok'] = df['HujanBesok'].map(label_map)

    # Imputasi median untuk kolom numerik jika diminta
    if impute_median:
        numeric_cols = ['Suhu', 'Kelembapan']
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Pisahkan fitur dan target
    X = df.drop(columns=['HujanBesok'])
    y = df['HujanBesok']
    return X, y


def _train_and_evaluate(
    algorithm: str,
    X,
    y,
    test_size: float,
    random_state: int,
    n_estimators: int,
    max_depth: int,
    max_features=None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    bootstrap: bool = True,
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Inisialisasi model berdasarkan pilihan algoritma
    if algorithm == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            max_features=max_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            bootstrap=bootstrap,
        )
    elif algorithm == 'logistic_regression':
        model = LogisticRegression(max_iter=1000, random_state=random_state)
    elif algorithm == 'svc':
        model = SVC(probability=False, random_state=random_state)
    elif algorithm == 'knn':
        model = KNeighborsClassifier()
    else:
        raise ValueError('Algoritma tidak dikenali')

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    akurasi = accuracy_score(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred, target_names=['Tidak', 'Ya'])
    cm = confusion_matrix(y_test, y_pred)

    return model, (X_train, X_test, y_train, y_test), akurasi, cls_report, cm


def _fig_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_base64


@login_required
def dashboard(request):
    context = {}

    if request.method == 'POST':
        form = MLConfigForm(request.POST)
        if form.is_valid():
            algorithm = form.cleaned_data['algorithm']
            n_estimators = form.cleaned_data['n_estimators']
            max_depth = form.cleaned_data['max_depth']
            test_size = form.cleaned_data['test_size']
            random_state = form.cleaned_data['random_state']
            use_ohe = form.cleaned_data['use_ohe']
            impute_median = form.cleaned_data['impute_median']
            # RF advanced params
            max_features = form.cleaned_data['max_features']
            min_samples_split = form.cleaned_data['min_samples_split']
            min_samples_leaf = form.cleaned_data['min_samples_leaf']
            bootstrap = form.cleaned_data['bootstrap']

            # Dataset dan preprocessing
            df = _generate_dummy_dataset()
            X, y = _preprocess(df, use_ohe=use_ohe, impute_median=impute_median)

            # Pelatihan dan evaluasi
            model, split_data, akurasi, cls_report, cm = _train_and_evaluate(
                algorithm,
                X,
                y,
                test_size,
                random_state,
                n_estimators,
                max_depth,
                max_features=max_features,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                bootstrap=bootstrap,
            )

            # Confusion matrix heatmap
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Tidak', 'Ya'], yticklabels=['Tidak', 'Ya'])
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            cm_img = _fig_to_base64()

            # Feature importance jika tersedia
            fi_img = None
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = X.columns
                fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)
                plt.figure(figsize=(8, 5))
                sns.barplot(data=fi_df, x='importance', y='feature', orient='h', color='skyblue')
                plt.title('Feature Importance')
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                fi_img = _fig_to_base64()

            messages.success(request, f"Pelatihan selesai. Akurasi: {akurasi:.4f}")
            context.update({
                'form': form,
                'accuracy': f"{akurasi:.4f}",
                'classification_report': cls_report,
                'cm_img': cm_img,
                'fi_img': fi_img,
                'algorithm': algorithm,
                'config_summary': {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'max_features': max_features if max_features is not None else 'auto',
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap,
                }
            })
        else:
            context['form'] = form
    else:
        context['form'] = MLConfigForm()

    return render(request, 'mlapp/index.html', context)


def home(request):
    return redirect('dashboard')


@login_required
def dataset(request):
    # Buat dataset dummy yang sama seperti untuk training
    df = _generate_dummy_dataset(n_rows=36)
    # Tampilkan sebagai tabel HTML dengan kelas Bootstrap
    table_html = df.to_html(classes=['table', 'table-striped', 'table-sm'], index=False)
    return render(request, 'mlapp/dataset.html', {
        'table_html': table_html
    })
