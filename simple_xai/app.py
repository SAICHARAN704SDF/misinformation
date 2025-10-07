"""PHASE 2: Tkinter application for misinformation detection + placeholder XAI.
Run AFTER training: python main.py
"""
import tkinter as tk
from tkinter import messagebox, scrolledtext
import joblib, os, json
import numpy as np
from simple_xai.main import preprocess_text, get_xai_explanation  # reuse functions

# --- Load artifacts ---
ART_DIR = 'artifacts'
MODEL_PATH = os.path.join(ART_DIR,'best_model.pkl')
VEC_PATH = os.path.join(ART_DIR,'tfidf_vectorizer.pkl')
if not (os.path.isfile(MODEL_PATH) and os.path.isfile(VEC_PATH)):
    print('Artifacts not found. Please run main.py to train.')
    raise SystemExit(1)
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)

# --- Prediction logic ---
THRESH_UNKNOWN = 0.5  # <= 50% -> "I don't know"

def predict():
    text = text_input.get('1.0', 'end-1c').strip()
    if not text:
        messagebox.showwarning('Input Error','Please enter text to analyze.')
        return
    clean = preprocess_text(text)
    vec = vectorizer.transform([clean])
    # Probability (if model lacks predict_proba we approximate)
    if hasattr(model,'predict_proba'):
        prob = model.predict_proba(vec)[0][1]
    else:
        # fallback logistic from decision_function
        decision = model.decision_function(vec)
        prob = float(1/(1+np.exp(-decision)))
    if prob <= THRESH_UNKNOWN:
        label_display = 'I do not know'
    else:
        label_display = 'Misleading' if prob >= 0.5 else 'Not Misleading'
    result_label.config(text=f'Misinformation Score: {prob*100:.2f}% -> {label_display}')
    # XAI placeholder
    explanation = get_xai_explanation(clean)
    xai_display.config(state=tk.NORMAL)
    xai_display.delete('1.0', tk.END)
    xai_display.insert(tk.END, 'Explanation (placeholder highlighting):\n')
    xai_display.insert(tk.END, explanation)
    xai_display.config(state=tk.DISABLED)
    feedback_frame.pack(pady=10)
    global last_prediction
    last_prediction = {'text': text,'clean': clean,'prob': prob,'label': label_display}

# --- Feedback ---
def submit_feedback(is_correct: bool):
    os.makedirs('feedback', exist_ok=True)
    path = os.path.join('feedback','feedback.csv')
    import csv, datetime
    with open(path,'a',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        if f.tell()==0:
            w.writerow(['text','clean_text','pred_prob','pred_label','correct','timestamp'])
        w.writerow([last_prediction.get('text',''), last_prediction.get('clean',''), last_prediction.get('prob',''), last_prediction.get('label',''), is_correct, datetime.datetime.utcnow().isoformat()])
    messagebox.showinfo('Feedback','Thank you! Feedback recorded.')
    feedback_frame.pack_forget()

# --- UI ---
root = tk.Tk()
root.title('Simple Misinformation Detector')
root.geometry('700x650')

title = tk.Label(root, text='Multilingual Misinformation Detector (Simplified)', font=('Segoe UI',16,'bold'))
title.pack(pady=10)

text_input = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=10, font=('Segoe UI',12))
text_input.pack(padx=10, pady=10)

predict_btn = tk.Button(root, text='Predict', font=('Segoe UI',12,'bold'), command=predict)
predict_btn.pack(pady=5)

result_label = tk.Label(root, text='Result will appear here', font=('Segoe UI', 13))
result_label.pack(pady=10)

xai_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=8, font=('Consolas',11), state=tk.DISABLED)
xai_display.pack(padx=10, pady=10)

feedback_frame = tk.Frame(root)
fb_label = tk.Label(feedback_frame, text='Is the label correct?', font=('Segoe UI', 12))
fb_label.pack(side=tk.LEFT, padx=5)
btn_yes = tk.Button(feedback_frame, text='Yes', command=lambda: submit_feedback(True))
btn_yes.pack(side=tk.LEFT, padx=5)
btn_no = tk.Button(feedback_frame, text='No', command=lambda: submit_feedback(False))
btn_no.pack(side=tk.LEFT, padx=5)

root.mainloop()
