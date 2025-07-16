# generate_plot_v3.py
from pymongo import MongoClient
from bson import ObjectId
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from collections import defaultdict
import os
import io
import base64
import time

def generate_plot(team_name="NinjaEnglineers"):
    start_time = time.time()
    os.environ["LOKY_MAX_CPU_COUNT"] = "8"
    load_dotenv()
    MONGO_URI = os.getenv("MONGO_URI")
    DB_NAME = os.getenv("DB_NAME")
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    # --- Fonctions utilitaires ---
    def extract_big5_vector(tests):
        if not tests or not tests.get("questions"):
            return None
        questions = tests.get("_sorted_questions")
        if not questions:
            questions = sorted(tests.get("questions", []), key=lambda q: int(q.get("questionId", 0)))
            tests["_sorted_questions"] = questions
        vector = []
        for q in questions:
            try:
                answer = int(q.get("answers", [0])[0])
            except (ValueError, TypeError):
                answer = 0
            vector.append(answer)
        return np.array(vector, dtype=float) if len(vector) >= 50 else None

    def get_big5_test(candidate):
        test_ids = [ObjectId(t.get("$oid")) if isinstance(t, dict) else t for t in candidate.get("tests", [])]
        tests = db.tests.find({"_id": {"$in": test_ids}}, {"name": 1, "questions": 1, "specId": 1})
        return next((t for t in tests if t.get("specId") == "003" or "big 5" in t.get("name", "").lower()), None)

    def find_top_candidates(X, team_vector, index_to_candidate, min_results=10, k_start=5, k_max=50):
        knn = NearestNeighbors(n_neighbors=min(k_max, len(X)), metric='euclidean').fit(X)
        distances, indices = knn.kneighbors([team_vector])
        candidates_near = [(index_to_candidate[idx], distances[0][i]) for i, idx in enumerate(indices[0][:min_results])]
        return candidates_near if len(candidates_near) >= min_results else []

    def distance_to_score(distance, min_d=5, max_d=12):
        distance = np.clip(distance, min_d, max_d)
        return round(np.clip(100 - ((distance - min_d) / (max_d - min_d)) * 40, 0, 100), 1)

    # --- Charger l’équipe cible ---
    t1 = time.time()
    team = db.teams.find_one({"name": team_name}, {"name": 1, "members": 1})
    if not team:
        print(f"Team query: {time.time() - t1:.2f}s")
        print(f"Total time: {time.time() - start_time:.2f}s")
        return None, f"Équipe '{team_name}' introuvable."

    member_ids = team.get("members", [])
    members = list(db.candidates.find(
        {"_id": {"$in": member_ids}},
        {"firstName": 1, "lastName": 1, "tests": 1, "_id": 1}
    ))
    print(f"Team and members query: {time.time() - t1:.2f}s")

    # --- Calculer le vecteur moyen de l’équipe ---
    t2 = time.time()
    member_vectors = []
    all_test_ids = set(t for m in members for t in m.get("tests", []))
    test_docs = {t["_id"]: t for t in db.tests.find(
        {"_id": {"$in": list(all_test_ids)}},
        {"name": 1, "questions": 1, "specId": 1}
    )}
    print(f"Member tests query: {time.time() - t2:.2f}s")

    t3 = time.time()
    for m in members:
        m_tests = [test_docs.get(ObjectId(t.get("$oid") if isinstance(t, dict) else t)) for t in m.get("tests", [])]
        big5 = next((t for t in m_tests if t and (t.get("specId") == "003" or "big 5" in t.get("name", "").lower())), None)
        vector = extract_big5_vector(big5)
        if vector is not None:
            member_vectors.append(vector)
    print(f"Member vector extraction: {time.time() - t3:.2f}s")

    if not member_vectors:
        print(f"Total time: {time.time() - start_time:.2f}s")
        return None, "Aucun membre avec un test Big 5 valide."

    team_vector = np.mean(np.stack(member_vectors), axis=0)

    # --- Charger tous les candidats autres que les membres ---
    t4 = time.time()
    excluded_ids = [m["_id"] for m in members]
    candidates = list(db.candidates.find(
        {"_id": {"$nin": excluded_ids}},
        {"firstName": 1, "lastName": 1, "tests": 1, "_id": 1}
    ))
    test_ids = set(t for c in candidates for t in c.get("tests", []))
    test_docs = {t["_id"]: t for t in db.tests.find(
        {"_id": {"$in": list(test_ids)}},
        {"name": 1, "questions": 1, "specId": 1}
    )}
    print(f"Candidates and tests query: {time.time() - t4:.2f}s")

    # --- Extraction des vecteurs candidats ---
    t5 = time.time()
    all_vectors = []
    valid_candidates = []

    for c in candidates:
        c_tests = [test_docs.get(ObjectId(t.get("$oid") if isinstance(t, dict) else t)) for t in c.get("tests", [])]
        big5 = next((t for t in c_tests if t and (t.get("specId") == "003" or "big 5" in t.get("name", "").lower())), None)
        vector = extract_big5_vector(big5)
        if vector is not None:
            all_vectors.append(vector)
            valid_candidates.append(c)
    print(f"Candidate vector extraction: {time.time() - t5:.2f}s")

    if not all_vectors:
        print(f"Total time: {time.time() - start_time:.2f}s")
        return None, "Aucun candidat trouvé."

    X = np.array(all_vectors)
    index_to_candidate = {i: c for i, c in enumerate(valid_candidates)}

    # --- Recherche KNN ---
    t6 = time.time()
    top_candidates_with_distances = find_top_candidates(X, team_vector, index_to_candidate)
    print(f"KNN computation: {time.time() - t6:.2f}s")

    if not top_candidates_with_distances:
        print(f"Total time: {time.time() - start_time:.2f}s")
        return None, "Aucun candidat compatible trouvé."

    # --- Résultats formatés ---
    results = []
    for c, dist in top_candidates_with_distances[:10]:
        score = distance_to_score(dist)
        name = f"{c.get('firstName', '')} {c.get('lastName', '')}".strip() or "Anonyme"
        results.append((name, dist, score))

    df_plot = pd.DataFrame({
        "candidate": [r[0] for r in results],
        "distance": [r[1] for r in results],
        "score": [r[2] for r in results]
    })

    # --- Visualisation ---
    # --- Visualisation ---
    t7 = time.time()
    plt.style.use('default')
    fig, (ax_bar, ax_radial) = plt.subplots(1, 2, figsize=(18, 9), dpi=100)
    fig.patch.set_facecolor('#F5F7FA')

    # Radial plot
    ax_radial.set_xlim(-30, 30)
    ax_radial.set_ylim(-30, 30)
    ax_radial.set_aspect('equal')
    ax_radial.set_title(f"Candidats proches de l’équipe '{team_name}'", fontsize=14)
    ax_radial.set_facecolor('#E6F0FA')
    ax_radial.grid(False)

    colors = ['gold', 'red', 'blue', 'green', 'purple', 'cyan', 'magenta', 'lime', 'pink', 'orange']
    angles = np.linspace(0, 2 * np.pi, len(df_plot), endpoint=False)
    center = np.array([0, 0])
    radii = np.linspace(5, 25, len(df_plot))

    for i, row in df_plot.iterrows():
        r = radii[i]
        angle = angles[i]
        x = center[0] + r * np.cos(angle)
        y = center[1] + r * np.sin(angle)
        color = colors[i % len(colors)]

        ax_radial.add_artist(plt.Circle(center, r, color=color, linestyle='--', fill=False, linewidth=1.2, alpha=0.5))
        ax_radial.scatter(x, y, s=100 - i * 5, color=color, marker='X', alpha=0.7, edgecolors='black', linewidth=0.5)
        ax_radial.annotate(f"{i+1}. {row['candidate'][:20]}",
                           (x, y), xytext=(x + 3 * np.sign(x), y + 3 * np.sign(y)),
                           fontsize=9, color='black',
                           bbox=dict(facecolor=color, alpha=0.3, edgecolor='none'),
                           arrowprops=dict(arrowstyle='->', color='black'))

    ax_radial.plot(0, 0, marker='x', color='black', markersize=12, mew=2)
    ax_radial.text(0, 1, f"Équipe: {team_name}", fontsize=9, ha='center', va='bottom', color='black')
    ax_radial.set_xticks([])
    ax_radial.set_yticks([])

    # Bar chart
    candidates_ranked = df_plot["candidate"][::-1]
    scores_ranked = df_plot["score"][::-1]
    bar_colors = colors[:len(candidates_ranked)][::-1]

    y_pos = np.arange(len(candidates_ranked))
    bars = ax_bar.barh(y_pos, scores_ranked, height=0.6, color=bar_colors, edgecolor='black')

    for i, bar in enumerate(bars):
        score = scores_ranked.iloc[i]
        ax_bar.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{score:.1f}%", va='center', fontsize=9)

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(candidates_ranked, fontsize=9)
    ax_bar.set_xlim(0, 110)
    ax_bar.set_xlabel("Taux de compatibilité (%)", fontsize=11)
    ax_bar.set_title("Top 10 candidats pour l’équipe", fontsize=13, pad=10)
    ax_bar.grid(axis='x', linestyle='--', alpha=0.3)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['left'].set_color('gray')
    ax_bar.spines['bottom'].set_color('gray')

    plt.tight_layout()

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    print(f"Plotting: {time.time() - t7:.2f}s")
    print(f"Total time: {time.time() - start_time:.2f}s")
    return img_str, None
