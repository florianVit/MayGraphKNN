from flask import Flask, render_template, request, jsonify


from pymongo import MongoClient
from dotenv import load_dotenv
import os
import time
from bson import ObjectId



app = Flask(__name__)

# Initialize MongoDB
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# In-memory cache: { (input_id, version): (img_str, timestamp) }
cache = {}
CACHE_DURATION = 3600  # Cache results for 1 hour

@app.route('/')
def index():
    print("Accessing the / route")
    return render_template('index.html')

@app.route('/teams', methods=['GET'])
def get_teams():
    try:
        teams = list(db.teams.find({}, {"name": 1, "_id": 0}))
        return jsonify([team['name'] for team in teams])
    except Exception as e:
        print(f"Error fetching teams: {e}")
        return jsonify({'error': 'Error fetching teams.'}), 500

@app.route('/candidates', methods=['GET'])
def get_candidates():
    try:
        candidates = list(db.candidates.find({}, {"firebaseId": 1, "firstName": 1, "lastName": 1, "_id": 0}))
        return jsonify([
            {
                'firebaseId': c['firebaseId'],
                'firstName': c.get('firstName', ''),
                'lastName': c.get('lastName', '')
            } for c in candidates
        ])
    except Exception as e:
        print(f"Error fetching candidates: {e}")
        return jsonify({'error': 'Error fetching candidates.'}), 500

@app.route('/team-members', methods=['GET'])
def get_team_members():
    try:
        team_name = request.args.get('name', '').strip()
        if not team_name:
            return jsonify({'error': 'No team name provided.'}), 400

        team = db.teams.find_one({"name": team_name}, {"members": 1})
        if not team:
            return jsonify({'error': f"Team '{team_name}' not found."}), 404

        member_ids = [ObjectId(m) for m in team.get('members', [])]
        members = list(db.candidates.find(
            {"_id": {"$in": member_ids}},
            {"firstName": 1, "lastName": 1}
        ))
        member_names = [
            f"{m.get('firstName', '')} {m.get('lastName', '')}".strip() or "Anonymous"
            for m in members
        ]
        return jsonify({'members': member_names})
    except Exception as e:
        print(f"Error fetching team members: {e}")
        return jsonify({'error': 'Error fetching team members.'}), 500

@app.route('/candidate-team', methods=['GET'])
def get_candidate_team():
    try:
        firebase_id = request.args.get('firebaseId', '').strip()
        if not firebase_id:
            return jsonify({'error': 'No candidate ID provided.'}), 400

        candidate = db.candidates.find_one({"firebaseId": firebase_id}, {"_id": 1})
        if not candidate:
            return jsonify({'error': f"Candidate '{firebase_id}' not found."}), 404

        team = db.teams.find_one({"members": candidate['_id']}, {"name": 1})
        return jsonify({'team': team['name'] if team else None})
    except Exception as e:
        print(f"Error fetching candidate team: {e}")
        return jsonify({'error': 'Error fetching candidate team status.'}), 500

@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    cache.clear()
    print("Cache cleared")
    return jsonify({'message': 'Cache cleared successfully'})

@app.route('/generate', methods=['POST'])
def generate():
    from big5 import generate_plot as generate_plot_v1
    from personality import generate_plot as generate_plot_v2  


    from big5team import generate_plot as generate_plot_v3
    from personalityteam import generate_plot as generate_plot_v4  


    print("Received POST request on /generate")
    input_id = request.form.get('input_id', '').strip()
    version = request.form.get('version', 'big5')
    print(f"Processing with input_id: {input_id}, version: {version}")

    if not input_id:
        return jsonify({'error': 'No candidate ID or team name provided.'}), 400

    # Check cache
    cache_key = (input_id, version)
    current_time = time.time()
    if cache_key in cache:
        img_str, timestamp = cache[cache_key]
        if current_time - timestamp < CACHE_DURATION:
            print(f"Returning cached result for {cache_key}")
            return jsonify({'image': img_str})

    # Determine if input_id is a firebase_id or team name
    candidate = db.candidates.find_one({"firebaseId": input_id}, {"_id": 1})
    team = db.teams.find_one({"name": input_id}, {"_id": 1})

    # Map version to script
    if candidate:
        if version == 'big5':
            generate_function = generate_plot_v1
        elif version == 'personality':
            generate_function = generate_plot_v2
        else:
            return jsonify({'error': f"Version '{version}' is not valid for a candidate."}), 400
    elif team:
        if version == 'big5':
            generate_function = generate_plot_v3
        elif version == 'personality':
            generate_function = generate_plot_v4
        else:
            return jsonify({'error': f"Version '{version}' is not valid for a team."}), 400
    else:
        return jsonify({'error': f"'{input_id}' is neither a valid candidate ID nor a team name."}), 400

    img_str, error = generate_function(input_id)
    if error:
        print(f"Error in generate_plot: {error}")
        return jsonify({'error': error}), 400

    # Store in cache
    cache[cache_key] = (img_str, current_time)
    print(f"Plot generated and cached for {cache_key}")
    return jsonify({'image': img_str})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

