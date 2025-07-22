# generate_plot_v2.py
from pymongo import MongoClient
from bson import ObjectId
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from collections import defaultdict
import os
import random
import io
import base64
import time

def generate_plot(firebase_id="f5qE4BSXEghVQtRHYeYoIMoU5lH2"):
    os.environ["LOKY_MAX_CPU_COUNT"] = "8"

    # --- Initialization ---
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

    # --- Extract combined vector ---
    def extract_combined_vector(tests, required_names):
        vector = []
        for name in required_names:
            test = tests.get(name)
            if not test:
                return None
            questions = sorted(test.get("questions", []), key=lambda q: int(q.get("questionId", 0)))
            if len(questions) == 0:
                return None
            for q in questions:
                raw_answer = q.get("answers", [None])[0]
                if raw_answer is None:
                    answer = 0
                elif TEST_TYPES[name] == "bool":
                    answer = 1 if isinstance(raw_answer, str) and raw_answer.endswith("-true") else 0
                else:
                    try:
                        answer = int(raw_answer)
                    except (ValueError, TypeError):
                        answer = 0
                vector.append(answer)
        return np.array(vector, dtype=float)

    # --- Load candidate ---
    candidate = db.candidates.find_one({"firebaseId": firebase_id})
    if not candidate:
        return None, "Candidate not found."

    test_ids = [ObjectId(t.get("$oid")) if isinstance(t, dict) else t for t in candidate.get("tests", [])]
    tests_cursor = db.tests.find({"_id": {"$in": test_ids}})
    candidate_tests = {t.get("name"): t for t in tests_cursor}

    candidate_vector = extract_combined_vector(candidate_tests, TEST_NAMES)
    if candidate_vector is None:
        return None, "Candidate has not completed all required questionnaires."

    # --- Load teams and members ---
    teams = list(db.teams.find({}))

    current_team = next((team.get("name") for team in teams if candidate["_id"] in team.get("members", [])), None)
    
    team_of_candidate = next((team for team in teams if candidate["_id"] in team.get("members", [])), None)
    if team_of_candidate:
        print(f"Candidate is already in the team: {team_of_candidate.get('name', 'Unknown name')}")

    member_ids = [m for team in teams for m in team.get("members", [])]
    members = list(db.candidates.find({"_id": {"$in": member_ids}}))

    all_test_ids = [t for m in members for t in m.get("tests", [])]
    test_docs = {t["_id"]: t for t in db.tests.find({"_id": {"$in": all_test_ids}})}

    all_vectors = []
    all_candidates = []
    team_lookup = {}

    for member in members:
        member_tests_ids = member.get("tests", [])
        member_tests = {}
        for t in member_tests_ids:
            test_obj = test_docs.get(ObjectId(t.get("$oid")) if isinstance(t, dict) else t)
            if test_obj and test_obj.get("name") in TEST_NAMES:
                member_tests[test_obj.get("name")] = test_obj
        vector = extract_combined_vector(member_tests, TEST_NAMES)
        if vector is not None:
            all_vectors.append(vector)
            all_candidates.append(member)
            for team in teams:
                if member["_id"] in team.get("members", []):
                    team_lookup[str(member["_id"])] = team.get("name", "Unknown team")
                    break

    # --- Align vectors ---
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
    candidate_vector = align_vectors([candidate_vector])[0]

    # --- Adaptive KNN ---
    def find_top_teams(X, candidate_vector, index_to_team, min_teams=10, k_start=5, k_max=50):
        for k in range(k_start, min(k_max, len(X)) + 1):
            knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
            knn.fit(X)
            distances, indices = knn.kneighbors([candidate_vector])
            team_members = defaultdict(list)
            for dist, idx in zip(distances[0], indices[0]):
                team = index_to_team.get(idx)
                if team:
                    team_members[team].append(dist)
            if len(team_members) >= min_teams:
                return team_members
        return {}

    index_to_candidate = {i: c for i, c in enumerate(all_candidates)}
    index_to_team = {i: team_lookup.get(str(c["_id"])) for i, c in index_to_candidate.items()}

    team_members = find_top_teams(X, candidate_vector, index_to_team)
    if not team_members:
        return None, "No similar teams found."

    # Exclude the candidate's current team
    if current_team:
        team_members = {team: dists for team, dists in team_members.items() if team != current_team}

    if not team_members:
        return None, "No similar teams found after removing candidate's current team."

    # --- Scoring ---
    def distance_to_score(distance, min_d=5, max_d=12):
        distance = np.clip(distance, min_d, max_d)
        return round(np.clip(100 - ((distance - min_d) / (max_d - min_d)) * 40, 0, 100), 1)

    results = [(np.mean(dists), team, len(dists)) for team, dists in team_members.items()]
    results.sort()
    top_teams = results[:10]

    df_plot = pd.DataFrame({
        "team": [r[1] for r in top_teams],
        "distance": [r[0] for r in top_teams],
        "score": [distance_to_score(r[0]) for r in top_teams],
        "size": [r[2] for r in top_teams]
    })

    # --- Visualization ---
    plt.style.use('default')
    fig, (ax_bar, ax_radial) = plt.subplots(1, 2, figsize=(18, 9), dpi=100)
    fig.patch.set_facecolor('#F5F7FA')

    # Radial plot
    ax_radial.set_xlim(-30, 30)
    ax_radial.set_ylim(-30, 30)
    ax_radial.set_aspect('equal')
    ax_radial.set_title("Teams closest to the candidate", fontsize=14)
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
        ax_radial.annotate(f"{i+1}. {row['team']}",
                           (x, y), xytext=(x + 3 * np.sign(x), y + 3 * np.sign(y)),
                           fontsize=9, color='black',
                           bbox=dict(facecolor=color, alpha=0.3, edgecolor='none'),
                           arrowprops=dict(arrowstyle='->', color='black'))

    ax_radial.plot(0, 0, marker='x', color='black', markersize=12, mew=2)
    ax_radial.text(0, 1, f"{candidate.get('firstName', '')} {candidate.get('lastName', '')}",
                   fontsize=9, ha='center', va='bottom', color='black')
    ax_radial.set_xticks([])
    ax_radial.set_yticks([])

    # Bar chart
    teams_ranked = df_plot["team"][::-1]
    scores_ranked = df_plot["score"][::-1]
    sizes_ranked = df_plot["size"][::-1]
    bar_colors = colors[:len(teams_ranked)][::-1]

    y_pos = np.arange(len(teams_ranked))
    bars = ax_bar.barh(y_pos, scores_ranked, height=0.6, color=bar_colors, edgecolor='black')

    for i, bar in enumerate(bars):
        score = scores_ranked.iloc[i]
        size = sizes_ranked.iloc[i]
        ax_bar.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{score:.1f}%  (from {size} member{'s' if size > 1 else ''})", va='center', fontsize=9)

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(teams_ranked, fontsize=9)
    ax_bar.set_xlim(0, 110)
    ax_bar.set_xlabel("Compatibility Rate (%)", fontsize=11)
    ax_bar.set_title("Team Ranking", fontsize=13, pad=10)
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

    return img_str, None
