# generate_plot_v4.py for team
from pymongo import MongoClient
from bson import ObjectId
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
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

    TEST_TYPES = {
        "Assertiveness": "likert",
        "Creative or Analytical": "likert",
        "Intellectual Curiosity": "likert",
        "Entrepreneur": "bool",
        "Self-Motivation": "bool"
    }
    TEST_NAMES = list(TEST_TYPES.keys())

    def extract_combined_vector(tests, required_names):
        vector = []
        for name in required_names:
            test = tests.get(name)
            if not test or not test.get("questions"):
                return None
            questions = test.get("_sorted_questions")
            if not questions:
                questions = sorted(test.get("questions", []), key=lambda q: int(q.get("questionId", 0)))
                test["_sorted_questions"] = questions
            for q in questions:
                raw_answer = q.get("answers", [None])[0]
                if raw_answer is None:
                    answer = 0
                elif TEST_TYPES[name] == "bool":
                    answer = 1 if isinstance(raw_answer, str) and raw_answer.lower().endswith("-true") else 0
                else:
                    try:
                        answer = int(raw_answer)
                    except (ValueError, TypeError):
                        answer = 0
                vector.append(answer)
        return np.array(vector, dtype=float)

    # --- Load the target team ---
    t1 = time.time()
    team = db.teams.find_one({"name": team_name}, {"name": 1, "members": 1})
    if not team:
        print(f"Team query: {time.time() - t1:.2f}s")
        print(f"Total time: {time.time() - start_time:.2f}s")
        return None, f"Team '{team_name}' not found."

    member_ids = team.get("members", [])
    members = list(db.candidates.find(
        {"_id": {"$in": member_ids}},
        {"firstName": 1, "lastName": 1, "tests": 1, "_id": 1}
    ))
    print(f"Team and members query: {time.time() - t1:.2f}s")

    # Preload members' tests
    t2 = time.time()
    test_ids = set(t for m in members for t in m.get("tests", []))
    test_docs = {t["_id"]: t for t in db.tests.find(
        {"_id": {"$in": list(test_ids)}},
        {"name": 1, "questions": 1}
    )}
    print(f"Member tests query: {time.time() - t2:.2f}s")

    t3 = time.time()
    member_vectors = []
    for m in members:
        member_tests = {}
        for t in m.get("tests", []):
            test = test_docs.get(ObjectId(t.get("$oid") if isinstance(t, dict) else t))
            if test and test.get("name") in TEST_NAMES:
                member_tests[test.get("name")] = test
        vector = extract_combined_vector(member_tests, TEST_NAMES)
        if vector is not None:
            member_vectors.append(vector)
    print(f"Member vector extraction: {time.time() - t3:.2f}s")

    if not member_vectors:
        print(f"Total time: {time.time() - start_time:.2f}s")
        return None, "No valid member vector found."

    team_vector = np.mean(np.stack(member_vectors), axis=0)

    # --- Load non-member candidates ---
    t4 = time.time()
    excluded_ids = [m["_id"] for m in members]
    candidates = list(db.candidates.find(
        {"_id": {"$nin": excluded_ids}},
        {"firstName": 1, "lastName": 1, "tests": 1, "_id": 1}
    ))
    candidate_ids = set(t for c in candidates for t in c.get("tests", []))
    test_docs = {t["_id"]: t for t in db.tests.find(
        {"_id": {"$in": list(candidate_ids)}},
        {"name": 1, "questions": 1}
    )}
    print(f"Candidates and tests query: {time.time() - t4:.2f}s")

    t5 = time.time()
    valid_candidates = []
    all_vectors = []
    for c in candidates:
        candidate_tests = {}
        for t in c.get("tests", []):
            test = test_docs.get(ObjectId(t.get("$oid") if isinstance(t, dict) else t))
            if test and test.get("name") in TEST_NAMES:
                candidate_tests[test.get("name")] = test
        vector = extract_combined_vector(candidate_tests, TEST_NAMES)
        if vector is not None:
            valid_candidates.append(c)
            all_vectors.append(vector)
    print(f"Candidate vector extraction: {time.time() - t5:.2f}s")

    if not all_vectors:
        print(f"Total time: {time.time() - start_time:.2f}s")
        return None, "No valid candidate found."

    # --- Vector alignment ---
    t6 = time.time()
    def align_vectors(vectors):
        max_len = max(len(v) for v in vectors)
        aligned = []
        for v in vectors:
            if len(v) < max_len:
                padded = np.pad(v, (0, max_len - len(v)), 'constant')
            else:
                padded = v[:max_len]
            aligned.append(padded)
        return np.array(aligned)

    X = align_vectors(all_vectors)
    team_vector = align_vectors([team_vector])[0]
    index_to_candidate = {i: c for i, c in enumerate(valid_candidates)}
    print(f"Vector alignment: {time.time() - t6:.2f}s")

    # --- KNN search ---
    t7 = time.time()
    def find_top_candidates(X, team_vector, index_to_candidate, min_results=10, k_start=20, k_max=50):
        knn = NearestNeighbors(n_neighbors=min(k_max, len(X)), metric='euclidean').fit(X)
        distances, indices = knn.kneighbors([team_vector])
        results = [(index_to_candidate[idx], distances[0][i]) for i, idx in enumerate(indices[0][:min_results])]
        return results if len(results) >= min_results else []

    def distance_to_score(distance, min_d=5, max_d=12):
        distance = np.clip(distance, min_d, max_d)
        return round(np.clip(100 - ((distance - min_d) / (max_d - min_d)) * 40, 0, 100), 1)

    top_candidates_with_distances = find_top_candidates(X, team_vector, index_to_candidate)
    print(f"KNN computation: {time.time() - t7:.2f}s")

    if not top_candidates_with_distances:
        print(f"Total time: {time.time() - start_time:.2f}s")
        return None, "No close candidate found."

    # --- Results ---
    results = []
    for c, dist in top_candidates_with_distances[:10]:
        score = distance_to_score(dist)
        name = f"{c.get('firstName', '')} {c.get('lastName', '')}".strip() or "Anonymous"
        results.append((name, dist, score))

    df_plot = pd.DataFrame({
        "candidate": [r[0] for r in results],
        "distance": [r[1] for r in results],
        "score": [r[2] for r in results]
    })

    # --- Visualization ---
    t8 = time.time()
    plt.style.use('default')
    fig, (ax_bar, ax_radial) = plt.subplots(1, 2, figsize=(18, 9), dpi=100)
    fig.patch.set_facecolor('#F5F7FA')

    # Radial plot
    ax_radial.set_xlim(-30, 30)
    ax_radial.set_ylim(-30, 30)
    ax_radial.set_aspect('equal')
    ax_radial.set_title(f"Top matching candidates for team '{team_name}'", fontsize=14)
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
        ax_radial.annotate(f"{i+1}. {row['candidate']}",
                           (x, y), xytext=(x + 3 * np.sign(x), y + 3 * np.sign(y)),
                           fontsize=9, color='black',
                           bbox=dict(facecolor=color, alpha=0.3, edgecolor='none'),
                           arrowprops=dict(arrowstyle='->', color='black'))

    ax_radial.plot(0, 0, marker='x', color='black', markersize=12, mew=2)
    ax_radial.text(0, 1, f"Team: {team_name}", fontsize=9, ha='center', va='bottom', color='black')
    ax_radial.set_xticks([])
    ax_radial.set_yticks([])

    # Bar chart
    candidates_ranked = df_plot["candidate"][::-1]
    scores_ranked = df_plot["score"][::-1]
    sizes_ranked = [len(members)] * len(df_plot)  # Team size used for each score
    bar_colors = colors[:len(candidates_ranked)][::-1]

    y_pos = np.arange(len(candidates_ranked))
    bars = ax_bar.barh(y_pos, scores_ranked, height=0.6, color=bar_colors, edgecolor='black')

    for i, bar in enumerate(bars):
        score = scores_ranked.iloc[i]
        size = sizes_ranked[i]
        ax_bar.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{score:.1f}%  (based on {size} member{'s' if size > 1 else ''})", va='center', fontsize=9)

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(candidates_ranked, fontsize=9)
    ax_bar.set_xlim(0, 110)
    ax_bar.set_xlabel("Compatibility Score (%)", fontsize=11)
    ax_bar.set_title("Candidate Ranking", fontsize=13, pad=10)
    ax_bar.grid(axis='x', linestyle='--', alpha=0.3)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['left'].set_color('gray')
    ax_bar.spines['bottom'].set_color('gray')

    plt.tight_layout()

    # Encode image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    print(f"Plotting: {time.time() - t8:.2f}s")
    print(f"Total time: {time.time() - start_time:.2f}s")
    return img_str, None
