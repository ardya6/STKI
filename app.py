
from flask import Flask, request, render_template_string
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ==========================================
# 1. MEMUAT DATASET
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

def load_dataset():
    files = sorted(DATASET_DIR.glob("dataset-*.csv"))
    datasets = []

    if not files:
        raise FileNotFoundError(
            f"File CSV tidak ditemukan di {DATASET_DIR}"
        )

    for file in files:
        df = pd.read_csv(file)

        required_columns = ["Title", "Ingredients", "Steps", "Loves", "URL"]

        if not all(col in df.columns for col in required_columns):
            print(f"Melewati file dengan kolom tidak sesuai: {file.name}")
            continue

        df = df[required_columns].copy()

        # Kategori diambil dari nama file
        df["Category"] = (
            file.stem.replace("dataset-", "").capitalize()
        )

        datasets.append(df)

    if not datasets:
        raise ValueError("Tidak ada dataset dengan kolom yang sesuai.")

    data = pd.concat(datasets, ignore_index=True)

    # Menangani data kosong
    for col in ["Title", "Ingredients", "Steps", "URL"]:
        data[col] = data[col].fillna("").astype(str)

    return data


recipes = load_dataset()

# ==========================================
# 2. MEMBANGUN MODEL TF-IDF
# ==========================================

# Nama resep diberi bobot lebih besar daripada bahan
recipes["SearchText"] = (
    recipes["Title"] + " " +
    recipes["Title"] + " " +
    recipes["Ingredients"]
)

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    sublinear_tf=True,
    token_pattern=r"(?u)\b\w+\b"
)

tfidf_matrix = vectorizer.fit_transform(recipes["SearchText"])

print(f"Jumlah resep berhasil dimuat: {len(recipes)}")

# ==========================================
# 3. TAMPILAN WEBSITE
# ==========================================

HTML = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>STKI Resep Masakan</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
          rel="stylesheet">
    <style>
        body { background: #f5f7f5; }
        .hero { background: #176b45; color: white; padding: 35px 0; }
        .recipe-card { border: 0; border-radius: 12px; }
        .score { color: #176b45; font-weight: bold; }
        .ingredients, .steps {
            white-space: pre-line;
            overflow-wrap: anywhere;
        }
    </style>
</head>
<body>

<header class="hero">
    <div class="container">
        <h1>STKI Resep Masakan</h1>
        <p class="mb-0">
            Temukan resep berdasarkan bahan yang tersedia di rumah.
        </p>
    </div>
</header>

<main class="container py-4">
    <form method="GET" action="/" class="card card-body mb-4">
        <label for="q" class="form-label fw-bold">
            Masukkan bahan atau nama masakan
        </label>
        <input id="q" name="q" class="form-control mb-3"
               placeholder="Contoh: telur, nasi, bawang merah"
               value="{{ query }}">

        <label for="category" class="form-label">Kategori resep</label>
        <select id="category" name="category" class="form-select mb-3">
            <option value="">Semua kategori</option>
            {% for c in categories %}
            <option value="{{ c }}"
                {% if category == c %}selected{% endif %}>
                {{ c }}
            </option>
            {% endfor %}
        </select>

        <button type="submit" class="btn btn-success">
            Cari Resep
        </button>
    </form>

    {% if query %}
        <h2 class="h4 mb-3">Hasil pencarian</h2>
        <p>Ditemukan {{ results|length }} resep yang ditampilkan.</p>

        {% if results %}
            {% for r in results %}
            <article class="card recipe-card shadow-sm mb-3">
                <div class="card-body">
                    <span class="badge text-bg-success">
                        {{ r.category }}
                    </span>
                    <h3 class="h5 mt-2">{{ r.title }}</h3>

                    <p class="score">
                        Skor Cosine Similarity:
                        {{ "%.4f"|format(r.score) }}
                    </p>

                    <details>
                        <summary class="mb-2">Lihat bahan dan cara memasak</summary>
                        <h4 class="h6 mt-3">Bahan-bahan</h4>
                        <p class="ingredients">{{ r.ingredients }}</p>

                        <h4 class="h6">Langkah memasak</h4>
                        <p class="steps">{{ r.steps }}</p>
                    </details>
                </div>
            </article>
            {% endfor %}
        {% else %}
            <div class="alert alert-warning">
                Tidak ditemukan resep yang cocok.
                Coba kata pencarian atau kategori lain.
            </div>
        {% endif %}
    {% else %}
        <div class="text-center py-4">
            <h2 class="h4">Mulai mencari resep</h2>
            <p class="text-secondary">
                Masukkan bahan atau nama masakan pada kolom di atas.
            </p>
        </div>
    {% endif %}
</main>

</body>
</html>
"""

# ==========================================
# 4. PROSES PENCARIAN
# ==========================================

@app.route("/", methods=["GET"])
def home():
    query = request.args.get("q", "").strip()
    category = request.args.get("category", "").strip()

    categories = sorted(recipes["Category"].unique())
    results = []

    if query:
        # Query diubah menjadi vektor TF-IDF
        query_vector = vectorizer.transform([query])

        # Hitung kemiripan query dengan seluruh resep
        scores = cosine_similarity(
            query_vector, tfidf_matrix
        ).ravel()

        # Filter kategori jika dipilih
        eligible = recipes.index.to_list()

        if category in categories:
            eligible = recipes.index[
                recipes["Category"] == category
            ].to_list()

        # Urutkan berdasarkan skor tertinggi
        ranked_indices = sorted(
            eligible,
            key=lambda i: scores[i],
            reverse=True
        )

        # Hanya tampilkan resep dengan skor di atas nol
        ranked_indices = [
            i for i in ranked_indices if scores[i] > 0
        ][:20]

        for i in ranked_indices:
            row = recipes.iloc[i]

            results.append({
                "title": row["Title"],
                "ingredients": row["Ingredients"].replace("--", "\n"),
                "steps": row["Steps"].replace("--", "\n"),
                "category": row["Category"],
                "score": float(scores[i]),
            })

    return render_template_string(
        HTML,
        query=query,
        category=category,
        categories=categories,
        results=results
    )


if __name__ == "__main__":
    app.run(debug=True)

