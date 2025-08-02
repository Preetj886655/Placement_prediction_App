import numpy as np
import pickle
from flask import Flask, request, render_template

# Load models
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model1.pkl", "rb") as f:
    model1 = pickle.load(f)

app = Flask(__name__, template_folder="templates")

@app.route('/')
def h():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/index')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    # Collecting input values safely with default fallbacks
    def safe_get(param):
        return request.args.get(param, '0') or '0'

    cgpa = safe_get('cgpa')
    projects = safe_get('projects')
    workshops = safe_get('workshops')
    mini_projects = safe_get('mini_projects')
    skills = request.args.get('skills', '')
    communication_skills = safe_get('communication_skills')
    internship = safe_get('internship')
    hackathon = safe_get('hackathon')
    tw_percentage = safe_get('tw_percentage')
    te_percentage = safe_get('te_percentage')
    backlogs = safe_get('backlogs')
    name = request.args.get('name', 'Student')

    # Skill count using comma split
    skill_count = len(skills.split(',')) if skills else 0

    try:
        # First model prediction: Placement
        input_features = np.array([
            cgpa, projects, workshops, mini_projects, skill_count,
            communication_skills, internship, hackathon,
            tw_percentage, te_percentage, backlogs
        ], dtype=float)

        placement_output = model.predict([input_features])[0]
        
        # Check if model supports probability
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba([input_features])[0][1]
        else:
            prob = 0.5  # fallback if model doesn't support probability

        prob_percent = round(prob * 100, 2)

        # Predict salary using model1 with placement info
        placement_flag = 1 if placement_output == 'Placed' else 0
        salary_input = np.array([
            cgpa, projects, workshops, mini_projects, skill_count,
            communication_skills, internship, hackathon,
            tw_percentage, te_percentage, backlogs, placement_flag
        ], dtype=float)

        raw_salary = model1.predict([salary_input])[0]
        salary = int(raw_salary + 1000000)  # add 10 lakh

        # Format salary nicely with commas (e.g., 12,00,000)
        def format_indian_currency(num):
            s = str(int(num))
            if len(s) <= 3:
                return s
            last3 = s[-3:]
            rest = s[:-3]
            rest = list(rest[::-1])
            parts = [rest[i:i+2] for i in range(0, len(rest), 2)]
            return ','.join([''.join(p[::-1]) for p in parts][::-1]) + ',' + last3

        formatted_salary = format_indian_currency(salary)
        formatted_salary = "₹" + format_indian_currency(salary)


        # Create message based on placement result
        if placement_output == 'Placed':
            out = f'Congratulations {name}! You have high chances of getting placed!'
            out2 = f'Your Expected Salary is INR {formatted_salary} per annum.'
        else:
            out = f'Sorry {name}, you currently have low chances of placement. Keep working hard!'
            out2 = 'Tip: Improve your technical and soft skills.'

        out3 = f'Probability of Placement: {prob_percent}%'

        return render_template('output.html', output=out, output2=out2, output3=out3)

    except Exception as e:
        return render_template('output.html', output="Error occurred!", output2=str(e), output3='')

if __name__ == "__main__":
    app.run(debug=True)
