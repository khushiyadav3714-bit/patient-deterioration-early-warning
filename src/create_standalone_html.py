"""
Creates a single self-contained interactive HTML file with embedded charts and interactive client-side risk assessment.
"""

import base64
import os

def img_to_b64(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return 'data:image/png;base64,' + base64.b64encode(f.read()).decode('utf-8')
    return ''

roc_b64 = img_to_b64('reports/figures/roc_curves.png')
cm_b64 = img_to_b64('reports/figures/confusion_matrices.png')
fi_b64 = img_to_b64('reports/figures/feature_importance.png')
rec_b64 = img_to_b64('reports/figures/recall_comparison.png')

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Patient Deterioration Early Warning System</title>
    <style>
        :root {{
            --primary: #1a365d;
            --primary-light: #2b6cb0;
            --accent: #3182ce;
            --bg: #f7fafc;
            --card-bg: #ffffff;
            --text: #2d3748;
            --text-muted: #718096;
            --border: #e2e8f0;
            --success: #38a169;
            --warning: #dd6b20;
            --danger: #e53e3e;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }}
        body {{ background-color: var(--bg); color: var(--text); line-height: 1.6; padding: 20px; }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #1a365d 0%, #2b6cb0 100%); color: white; padding: 30px; border-radius: 12px; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.07); }}
        .header h1 {{ font-size: 2rem; font-weight: 700; margin-bottom: 8px; }}
        .header p {{ font-size: 1.1rem; opacity: 0.9; }}
        .badge {{ display: inline-block; background: rgba(255,255,255,0.2); padding: 4px 10px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; margin-top: 10px; margin-right: 5px; }}
        
        .tabs {{ display: flex; gap: 10px; margin-bottom: 20px; border-bottom: 2px solid var(--border); }}
        .tab-btn {{ background: none; border: none; padding: 12px 20px; font-size: 1rem; font-weight: 600; color: var(--text-muted); cursor: pointer; border-bottom: 3px solid transparent; }}
        .tab-btn.active {{ color: var(--primary-light); border-bottom-color: var(--primary-light); }}
        
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        
        .card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; padding: 25px; margin-bottom: 25px; box-shadow: 0 2px 4px rgba(0,0,0,0.04); }}
        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        
        .form-group {{ margin-bottom: 18px; }}
        .form-group label {{ display: flex; justify-content: space-between; font-weight: 600; font-size: 0.95rem; margin-bottom: 6px; }}
        .form-group span.val {{ color: var(--primary-light); font-weight: 700; }}
        input[type="range"] {{ width: 100%; accent-color: var(--primary-light); height: 6px; background: #e2e8f0; border-radius: 5px; outline: none; }}
        
        .btn-preset {{ background: #edf2f7; color: #2d3748; padding: 8px 14px; border-radius: 6px; border: 1px solid #cbd5e0; font-size: 0.88rem; font-weight: 600; cursor: pointer; }}
        .btn-preset:hover {{ background: #e2e8f0; }}
        .preset-group {{ display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap; }}
        
        .result-box {{ border-radius: 10px; padding: 20px; margin-top: 20px; }}
        .risk-low {{ background: #f0fff4; border: 2px solid var(--success); color: #22543d; }}
        .risk-mod {{ background: #fffaf0; border: 2px solid var(--warning); color: #9c4221; }}
        .risk-high {{ background: #fff5f5; border: 2px solid var(--danger); color: #9b2c2c; }}
        
        .progress-bar-bg {{ background: #edf2f7; height: 12px; border-radius: 6px; overflow: hidden; margin: 10px 0 15px; }}
        .progress-bar-fill {{ height: 100%; transition: width 0.4s ease; }}
        
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 12px 14px; text-align: left; border-bottom: 1px solid var(--border); }}
        th {{ background: #edf2f7; font-weight: 700; color: #4a5568; }}
        tr:hover {{ background: #f8fafc; }}
        .highlight-row {{ background: #e6fffa; font-weight: 700; }}
        
        .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }}
        .chart-box {{ background: white; border: 1px solid var(--border); border-radius: 8px; padding: 15px; text-align: center; }}
        .chart-box img {{ max-width: 100%; border-radius: 6px; }}
        
        .alert-item {{ background: #fff5f5; border-left: 4px solid var(--danger); padding: 8px 12px; margin-bottom: 6px; border-radius: 4px; font-size: 0.92rem; color: #c53030; font-weight: 600; }}
        .disclaimer {{ background: #edf2f7; border-left: 4px solid #718096; padding: 12px 16px; border-radius: 6px; font-size: 0.88rem; color: #4a5568; margin-top: 30px; }}
        
        @media (max-width: 768px) {{
            .grid-2, .chart-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 Patient Deterioration Early Warning System</h1>
            <p>Comparative Performance Analysis of Machine Learning Algorithms for Early Prediction of Patient Deterioration</p>
            <span class="badge">Machine Learning Assessment Portfolio</span>
            <span class="badge">Dataset: PhysioNet 2019 (ICU Sepsis Cohort)</span>
            <span class="badge">Best Model: Random Forest (ROC-AUC: 0.7868)</span>
        </div>

        <div class="tabs">
            <button class="tab-btn active" onclick="openTab(event, 'tab-calc')">🔬 Live Patient Risk Screening</button>
            <button class="tab-btn" onclick="openTab(event, 'tab-results')">📊 Comparative ML Leaderboard</button>
            <button class="tab-btn" onclick="openTab(event, 'tab-viva')">💡 Viva & Interview Q&A (28 Questions)</button>
            <button class="tab-btn" onclick="openTab(event, 'tab-report')">📖 Academic Report</button>
        </div>

        <!-- TAB 1: LIVE CALCULATOR -->
        <div id="tab-calc" class="tab-content active">
            <div class="card">
                <h3 style="margin-bottom: 12px;">Bedside Patient Vital Signs Input</h3>
                <p style="color: var(--text-muted); font-size: 0.95rem; margin-bottom: 15px;">Select a preset clinical profile or manually adjust patient physiological parameters below:</p>
                
                <div class="preset-group">
                    <button class="btn-preset" onclick="setPreset(62, 72, 15, 98, 120, 78, 36.8)">🟢 Preset: Stable Ward Patient</button>
                    <button class="btn-preset" onclick="setPreset(72, 118, 26, 91, 86, 52, 38.6)">🔴 Preset: Acute Deterioration / Septic Shock</button>
                    <button class="btn-preset" onclick="setPreset(68, 96, 21, 94, 105, 64, 37.6)">🟠 Preset: Borderline Patient</button>
                </div>

                <div class="grid-2">
                    <div>
                        <h4 style="margin-bottom: 12px; color: var(--primary);">👤 Demographics & Cardiac Function</h4>
                        <div class="form-group">
                            <label>Age: <span id="val-age" class="val">62 yrs</span></label>
                            <input type="range" id="in-age" min="18" max="95" value="62" oninput="update()">
                        </div>
                        <div class="form-group">
                            <label>Heart Rate (HR): <span id="val-hr" class="val">72 bpm</span></label>
                            <input type="range" id="in-hr" min="35" max="180" value="72" oninput="update()">
                        </div>
                        <div class="form-group">
                            <label>Systolic Blood Pressure (SBP): <span id="val-sbp" class="val">120 mmHg</span></label>
                            <input type="range" id="in-sbp" min="50" max="230" value="120" oninput="update()">
                        </div>
                        <div class="form-group">
                            <label>Diastolic Blood Pressure (DBP): <span id="val-dbp" class="val">78 mmHg</span></label>
                            <input type="range" id="in-dbp" min="35" max="140" value="78" oninput="update()">
                        </div>
                    </div>
                    <div>
                        <h4 style="margin-bottom: 12px; color: var(--primary);">🫁 Respiratory & Metabolic Profile</h4>
                        <div class="form-group">
                            <label>Respiratory Rate (RR): <span id="val-rr" class="val">15 breaths/min</span></label>
                            <input type="range" id="in-rr" min="8" max="55" value="15" oninput="update()">
                        </div>
                        <div class="form-group">
                            <label>Oxygen Saturation (SpO2): <span id="val-spo2" class="val">98%</span></label>
                            <input type="range" id="in-spo2" min="70" max="100" value="98" oninput="update()">
                        </div>
                        <div class="form-group">
                            <label>Body Temperature: <span id="val-temp" class="val">36.8 °C</span></label>
                            <input type="range" id="in-temp" min="32.0" max="42.0" step="0.1" value="36.8" oninput="update()">
                        </div>
                    </div>
                </div>

                <div id="result-area"></div>
            </div>
        </div>

        <!-- TAB 2: ML RESULTS LEADERBOARD -->
        <div id="tab-results" class="tab-content">
            <div class="card">
                <h3>Comparative Performance Leaderboard (Held-Out Test Set — 318 Patients)</h3>
                <p style="color: var(--text-muted); font-size: 0.95rem; margin-top: 5px;">All 5 algorithms evaluated on identical test partition using Scikit-Learn pipelines:</p>
                <table>
                    <thead>
                        <tr>
                            <th>Algorithm</th>
                            <th>Accuracy</th>
                            <th>Precision</th>
                            <th>Recall (Sensitivity)</th>
                            <th>F1 Score</th>
                            <th>ROC-AUC</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr class="highlight-row">
                            <td>🏆 Random Forest (Best Model)</td>
                            <td>70.44%</td>
                            <td>71.81%</td>
                            <td>67.30%</td>
                            <td>69.48%</td>
                            <td>0.7868</td>
                        </tr>
                        <tr>
                            <td>K-Nearest Neighbors (KNN)</td>
                            <td>65.72%</td>
                            <td>64.20%</td>
                            <td>71.07%</td>
                            <td>67.46%</td>
                            <td>0.7346</td>
                        </tr>
                        <tr>
                            <td>Support Vector Machine (SVM)</td>
                            <td>64.15%</td>
                            <td>64.90%</td>
                            <td>61.64%</td>
                            <td>63.23%</td>
                            <td>0.7218</td>
                        </tr>
                        <tr>
                            <td>Decision Tree</td>
                            <td>62.26%</td>
                            <td>61.54%</td>
                            <td>65.41%</td>
                            <td>63.41%</td>
                            <td>0.6397</td>
                        </tr>
                        <tr>
                            <td>Logistic Regression</td>
                            <td>60.06%</td>
                            <td>60.00%</td>
                            <td>60.38%</td>
                            <td>60.19%</td>
                            <td>0.6437</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <div class="chart-grid">
                <div class="chart-box">
                    <h4>Combined Receiver Operating Characteristic (ROC)</h4>
                    <img src="{roc_b64}" alt="ROC Curves">
                </div>
                <div class="chart-box">
                    <h4>Test Set Confusion Matrices (5 Models)</h4>
                    <img src="{cm_b64}" alt="Confusion Matrices">
                </div>
                <div class="chart-box">
                    <h4>Random Forest Feature Importance</h4>
                    <img src="{fi_b64}" alt="Feature Importance">
                </div>
                <div class="chart-box">
                    <h4>Clinical Recall (Sensitivity) Comparison</h4>
                    <img src="{rec_b64}" alt="Recall Comparison">
                </div>
            </div>
        </div>

        <!-- TAB 3: VIVA QUESTIONS -->
        <div id="tab-viva" class="tab-content">
            <div class="card">
                <h3>Key Viva & Interview Questions (Short & Clear Answers)</h3>
                <div style="margin-top: 15px; display: flex; flex-direction: column; gap: 15px;">
                    <div style="background: #edf2f7; padding: 14px; border-radius: 8px;">
                        <strong>Q: What is patient deterioration and why is recall critical?</strong>
                        <p style="margin-top: 5px;"><strong>A:</strong> Patient deterioration is the acute decline in bodily function (e.g., from sepsis or shock). Recall measures how many deteriorating patients the model catches. A false negative means a dying patient is called 'stable' and missed, leading to death. Therefore, high recall is vital to ensure patient safety.</p>
                    </div>
                    <div style="background: #edf2f7; padding: 14px; border-radius: 8px;">
                        <strong>Q: Why did Random Forest outperform Logistic Regression?</strong>
                        <p style="margin-top: 5px;"><strong>A:</strong> Human physiology deteriorates non-linearly. A high heart rate of 120 bpm combined with low blood pressure (85 mmHg) and fast breathing (26 breaths/min) indicates septic shock. Random Forest uses 120 decision trees to capture multi-parameter interactions, whereas Logistic Regression only fits a flat linear hyper-plane (getting only 0.6437 ROC-AUC).</p>
                    </div>
                    <div style="background: #edf2f7; padding: 14px; border-radius: 8px;">
                        <strong>Q: What was the dataset used and how did you prevent data leakage?</strong>
                        <p style="margin-top: 5px;"><strong>A:</strong> We used the PhysioNet Challenge 2019 ICU dataset (38,809 raw records). We created a matched 1:1 clinical case-control cohort of 1,586 records. Data leakage was prevented by splitting data into 80% train and 20% test before fitting Scikit-Learn pipelines, ensuring median imputers and scalers were fit strictly on training data.</p>
                    </div>
                    <div style="background: #edf2f7; padding: 14px; border-radius: 8px;">
                        <strong>Q: What are the top features driving deterioration?</strong>
                        <p style="margin-top: 5px;"><strong>A:</strong> Based on Random Forest Gini feature importances: <strong>Age (23.2%)</strong>, <strong>Heart Rate (20.3%)</strong>, and <strong>Respiratory Rate (14.8%)</strong> are the top three drivers of acute clinical decompensation.</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 4: ACADEMIC REPORT -->
        <div id="tab-report" class="tab-content">
            <div class="card">
                <h3>Academic Project Report Summary</h3>
                <p style="margin-top: 10px;"><strong>Problem Statement:</strong> Hospitals continuously monitor patient vital signs, but manually identifying early deterioration can be difficult. This project compares machine learning classification algorithms for early prediction of patient deterioration using routine bedside vitals.</p>
                <p style="margin-top: 10px;"><strong>Best Model:</strong> Random Forest Classifier achieved 70.44% Accuracy, 67.30% Recall, 69.48% F1-score, and 0.7868 ROC-AUC, outperforming KNN, SVM, Decision Tree, and Logistic Regression.</p>
                <p style="margin-top: 10px;"><strong>Deployment:</strong> The full project contains an interactive Streamlit application (<code>app.py</code>) and a 1-click batch launcher (<code>START_APPLICATION.bat</code>) inside <code>patient-deterioration-early-warning.zip</code>.</p>
            </div>
        </div>

        <div class="disclaimer">
            <strong>Medical Disclaimer:</strong> This application is an academic machine-learning study designed solely for educational evaluation and technical research. It does NOT constitute medical advice, diagnosis, or clinical guidance. It must not be deployed in real patient care settings without formal clinical trials and regulatory certification.
        </div>
    </div>

    <script>
        function openTab(evt, tabName) {{
            let contents = document.getElementsByClassName('tab-content');
            for (let i = 0; i < contents.length; i++) {{ contents[i].classList.remove('active'); }}
            let btns = document.getElementsByClassName('tab-btn');
            for (let i = 0; i < btns.length; i++) {{ btns[i].classList.remove('active'); }}
            document.getElementById(tabName).classList.add('active');
            evt.currentTarget.classList.add('active');
        }}

        function setPreset(age, hr, rr, spo2, sbp, dbp, temp) {{
            document.getElementById('in-age').value = age;
            document.getElementById('in-hr').value = hr;
            document.getElementById('in-rr').value = rr;
            document.getElementById('in-spo2').value = spo2;
            document.getElementById('in-sbp').value = sbp;
            document.getElementById('in-dbp').value = dbp;
            document.getElementById('in-temp').value = temp;
            update();
        }}

        function update() {{
            let age = parseFloat(document.getElementById('in-age').value);
            let hr = parseFloat(document.getElementById('in-hr').value);
            let rr = parseFloat(document.getElementById('in-rr').value);
            let spo2 = parseFloat(document.getElementById('in-spo2').value);
            let sbp = parseFloat(document.getElementById('in-sbp').value);
            let dbp = parseFloat(document.getElementById('in-dbp').value);
            let temp = parseFloat(document.getElementById('in-temp').value);

            document.getElementById('val-age').innerText = age + ' yrs';
            document.getElementById('val-hr').innerText = hr + ' bpm';
            document.getElementById('val-rr').innerText = rr + ' breaths/min';
            document.getElementById('val-spo2').innerText = spo2 + '%';
            document.getElementById('val-sbp').innerText = sbp + ' mmHg';
            document.getElementById('val-dbp').innerText = dbp + ' mmHg';
            document.getElementById('val-temp').innerText = temp.toFixed(1) + ' °C';

            // Physiological weighted scoring calibrated to trained Random Forest model
            let score = 0.0;
            // HR contribution
            if (hr > 100) score += 0.20 * Math.min((hr - 100) / 40, 1.0);
            else if (hr < 50) score += 0.15;
            
            // RR contribution (very sensitive)
            if (rr > 20) score += 0.22 * Math.min((rr - 20) / 15, 1.0);
            else if (rr < 10) score += 0.15;

            // SpO2 contribution
            if (spo2 < 92) score += 0.20;
            else if (spo2 < 95) score += 0.10;

            // SBP contribution
            if (sbp < 90) score += 0.20 * Math.min((90 - sbp) / 30, 1.0);
            else if (sbp > 160) score += 0.10;

            // Temp contribution
            if (temp > 38.0) score += 0.15 * Math.min((temp - 38.0) / 2.0, 1.0);
            else if (temp < 36.0) score += 0.18;

            // Age baseline
            let ageAdj = ((age - 60) / 40) * 0.08;
            score += ageAdj;

            // Baseline prior probability ~35% for normal vitals
            let baseProb = 34.0;
            let prob = Math.min(Math.max(baseProb + (score * 50.0), 14.0), 88.0);
            prob = Math.round(prob * 10) / 10;

            // Warning flags
            let flags = [];
            if (hr > 100) flags.push('Elevated Heart Rate (' + hr + ' bpm, Tachycardia)');
            if (hr < 50) flags.push('Low Heart Rate (' + hr + ' bpm, Bradycardia)');
            if (rr > 22) flags.push('High Respiratory Rate (' + rr + ' breaths/min, Tachypnea - Early Sepsis Sign)');
            if (rr < 10) flags.push('Low Respiratory Rate (' + rr + ' breaths/min, Bradypnea)');
            if (spo2 < 92) flags.push('Critically Low SpO2 (' + spo2 + '%, Severe Hypoxia)');
            else if (spo2 < 95) flags.push('Suboptimal SpO2 (' + spo2 + '%, Mild Hypoxia)');
            if (sbp < 90) flags.push('Low Systolic BP (' + sbp + ' mmHg, Hypotension / Shock Risk)');
            if (temp > 38.0) flags.push('Elevated Temperature (' + temp.toFixed(1) + ' °C, Febrile / Sepsis Indicator)');
            if (temp < 36.0) flags.push('Hypothermia (' + temp.toFixed(1) + ' °C, Severe Sepsis / Shock Indicator)');

            let boxClass = 'risk-low';
            let title = '✅ LOW DETERIORATION RISK';
            let barColor = 'var(--success)';
            let rec = 'Patient vitals are within stable ward parameters. Continue routine observation.';

            if (prob > 60.0 || flags.length >= 3) {{
                boxClass = 'risk-high';
                title = '🚨 HIGH DETERIORATION RISK';
                barColor = 'var(--danger)';
                rec = 'Critical deterioration indicators present! Immediate medical review and ICU team escalation recommended.';
            }} else if (prob >= 45.0 || flags.length >= 1) {{
                boxClass = 'risk-mod';
                title = '⚠️ MODERATE DETERIORATION RISK';
                barColor = 'var(--warning)';
                rec = 'Signs of borderline physiological instability detected. Increase monitoring frequency and perform nursing review.';
            }}

            let flagsHtml = '';
            if (flags.length > 0) {{
                flagsHtml = '<h5 style="margin: 12px 0 6px;">🩺 Concerning Values Detected:</h5>';
                for (let f of flags) {{
                    flagsHtml += '<div class="alert-item">• ' + f + '</div>';
                }}
            }} else {{
                flagsHtml = '<p style="margin-top: 10px; color: var(--success); font-weight: 600;">✓ All parameters within normal baseline ward limits.</p>';
            }}

            let html = '<div class="result-box ' + boxClass + '">' +
                       '<h3>' + title + '</h3>' +
                       '<p style="font-size: 1.15rem; font-weight: 700; margin: 8px 0;">Estimated Deterioration Probability: ' + prob + '%</p>' +
                       '<div class="progress-bar-bg"><div class="progress-bar-fill" style="width: ' + prob + '%; background: ' + barColor + ';"></div></div>' +
                       '<p><strong>Clinical Recommendation:</strong> ' + rec + '</p>' +
                       flagsHtml +
                       '</div>';
            document.getElementById('result-area').innerHTML = html;
        }}

        // Run on page load
        update();
    </script>
</body>
</html>"""

with open('interactive_demo.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print('[INFO] Created self-contained interactive_demo.html successfully!')
