from django import forms


ALGORITHM_CHOICES = [
    ("random_forest", "Random Forest"),
]


class MLConfigForm(forms.Form):
    # Pilihan algoritma
    algorithm = forms.ChoiceField(
        choices=ALGORITHM_CHOICES,
        initial="random_forest",
        label="Algoritma",
        widget=forms.Select(attrs={"class": "form-select", "disabled": True})
    )

    # Hyperparameter lanjutan RF
    max_features = forms.ChoiceField(
        choices=[("sqrt", "sqrt"), ("log2", "log2"), ("none", "Auto (None)")],
        initial="sqrt",
        label="Max Features (RF)",
        widget=forms.Select(attrs={"class": "form-select"})
    )
    min_samples_split = forms.IntegerField(
        min_value=2,
        max_value=100,
        initial=2,
        label="Min Samples Split (RF)",
        widget=forms.NumberInput(attrs={"class": "form-control", "min": 2, "max": 100})
    )
    min_samples_leaf = forms.IntegerField(
        min_value=1,
        max_value=100,
        initial=1,
        label="Min Samples Leaf (RF)",
        widget=forms.NumberInput(attrs={"class": "form-control", "min": 1, "max": 100})
    )
    bootstrap = forms.BooleanField(
        required=False,
        initial=True,
        label="Bootstrap (RF)",
        widget=forms.CheckboxInput(attrs={"class": "form-check-input"})
    )

    # Hyperparameter umum
    n_estimators = forms.IntegerField(
        min_value=10,
        max_value=1000,
        initial=100,
        label="N Estimators (RF)",
        widget=forms.NumberInput(attrs={"class": "form-control", "min": 10, "max": 1000})
    )
    max_depth = forms.IntegerField(
        min_value=1,
        max_value=100,
        initial=10,
        label="Max Depth (RF)",
        widget=forms.NumberInput(attrs={"class": "form-control", "min": 1, "max": 100})
    )

    # Pengaturan split
    test_size = forms.FloatField(
        min_value=0.05,
        max_value=0.5,
        initial=0.2,
        label="Test Size (0-1)",
        widget=forms.NumberInput(attrs={"class": "form-control", "step": 0.01})
    )
    random_state = forms.IntegerField(
        min_value=0,
        max_value=9999,
        initial=42,
        label="Random State",
        widget=forms.NumberInput(attrs={"class": "form-control"})
    )

    # Opsi pra-pemrosesan
    use_ohe = forms.BooleanField(
        required=False,
        initial=True,
        label="One-Hot Encoding KondisiAwan",
        widget=forms.CheckboxInput(attrs={"class": "form-check-input"})
    )
    impute_median = forms.BooleanField(
        required=False,
        initial=True,
        label="Imputasi Median Numerik",
        widget=forms.CheckboxInput(attrs={"class": "form-check-input"})
    )

    def clean(self):
        cleaned = super().clean()
        algo = cleaned.get("algorithm")
        n_estimators = cleaned.get("n_estimators")
        max_depth = cleaned.get("max_depth")
        test_size = cleaned.get("test_size")
        min_samples_split = cleaned.get("min_samples_split")
        min_samples_leaf = cleaned.get("min_samples_leaf")
        max_features = cleaned.get("max_features")

        # Validasi dasar
        if test_size and not (0.05 <= test_size <= 0.5):
            self.add_error("test_size", "Test size harus antara 0.05 dan 0.5")

        # Hanya relevan untuk Random Forest
        if algo == "random_forest":
            if n_estimators and n_estimators < 10:
                self.add_error("n_estimators", "Minimal 10 trees untuk Random Forest")
            if max_depth and max_depth < 1:
                self.add_error("max_depth", "Max depth harus >= 1")
            if min_samples_split and min_samples_split < 2:
                self.add_error("min_samples_split", "Min samples split minimal 2")
            if min_samples_leaf and min_samples_leaf < 1:
                self.add_error("min_samples_leaf", "Min samples leaf minimal 1")

        # Normalisasi nilai max_features 'none' -> None
        if isinstance(max_features, str) and max_features.lower() == "none":
            cleaned["max_features"] = None

        return cleaned