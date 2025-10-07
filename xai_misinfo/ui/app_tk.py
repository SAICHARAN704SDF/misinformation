import tkinter as tk
from tkinter import ttk, messagebox
import threading
from inference import InferenceEngine
from explain.lime_utils import lime_explain
from explain.shap_utils import explain_transformer
import json

class App:
    def __init__(self, root):
        self.root = root
        root.title('Multilingual Misinformation Detector (XAI)')
        self.engine = InferenceEngine()
        self._build_ui()
    def _build_ui(self):
        self.text_input = tk.Text(self.root, height=6, width=80)
        self.text_input.pack(padx=10, pady=10)
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=5)
        self.predict_btn = ttk.Button(btn_frame, text='Predict', command=self.predict_async)
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        self.explain_btn = ttk.Button(btn_frame, text='Explain', command=self.explain_async)
        self.explain_btn.pack(side=tk.LEFT, padx=5)
        self.feedback_yes = ttk.Button(btn_frame, text='Yes', command=lambda: self.record_feedback(True))
        self.feedback_no = ttk.Button(btn_frame, text='No', command=lambda: self.record_feedback(False))
        self.feedback_yes.pack(side=tk.LEFT, padx=5)
        self.feedback_no.pack(side=tk.LEFT, padx=5)
        self.result_var = tk.StringVar(value='Prediction:')
        ttk.Label(self.root, textvariable=self.result_var, font=('Segoe UI', 12, 'bold')).pack(pady=5)
        self.explain_box = tk.Text(self.root, height=12, width=80, bg='#f7f7f7')
        self.explain_box.pack(padx=10, pady=10)
    def predict_async(self):
        threading.Thread(target=self._predict, daemon=True).start()
    def _predict(self):
        txt = self.text_input.get('1.0', tk.END).strip()
        if not txt:
            messagebox.showwarning('Input required','Please enter text.')
            return
        out = self.engine.predict(txt)
        self.last_prediction = out
        self.result_var.set(f"Prediction: {out['label']}  (score={out['probability']*100:.2f}%)")
    def explain_async(self):
        threading.Thread(target=self._explain, daemon=True).start()
    def _explain(self):
        self.explain_box.delete('1.0', tk.END)
        txt = self.text_input.get('1.0', tk.END).strip()
        if not txt:
            return
        try:
            if self.engine.transformer:
                shap_values = explain_transformer(self.engine.transformer, self.engine.tokenizer, [txt], max_samples=1)
                # Display tokens with positive contribution
                data = []
                for sv in shap_values:
                    tokens = sv.data[0]
                    vals = sv.values[0]
                    for t,v in zip(tokens, vals):
                        if v>0:
                            data.append((t,float(v)))
                data = sorted(data, key=lambda x:x[1], reverse=True)[:15]
                self.explain_box.insert(tk.END, 'Top Positive Tokens (SHAP):\n')
                for t,v in data:
                    self.explain_box.insert(tk.END, f"{t} (+{v:.3f})\n")
            else:
                # Fallback lime using best classical model
                if not self.engine.vectorizer or not self.engine.classical:
                    self.explain_box.insert(tk.END, 'No model available for explanation.')
                    return
                # pick first classical model
                name, model = next(iter(self.engine.classical.items()))
                def predict_proba(texts):
                    vec = self.engine.vectorizer.transform(texts)
                    if hasattr(model,'predict_proba'):
                        return model.predict_proba(vec)
                    else:
                        import numpy as np
                        d = model.decision_function(vec)
                        if d.ndim==1:
                            probs = 1/(1+np.exp(-d))
                            return np.vstack([1-probs, probs]).T
                        return d
                exp = lime_explain(predict_proba, ['Not Misleading','Misleading'], txt)
                self.explain_box.insert(tk.END, 'LIME Explanation (tokens):\n')
                for w,score in exp.as_list():
                    self.explain_box.insert(tk.END, f"{w}: {score:+.3f}\n")
        except Exception as e:
            self.explain_box.insert(tk.END, f"Explanation failed: {e}")
    def record_feedback(self, correct: bool):
        # Append feedback CSV
        try:
            import csv, os, datetime
            os.makedirs('data', exist_ok=True)
            with open('data/feedback.csv','a',newline='',encoding='utf-8') as f:
                w = csv.writer(f)
                if f.tell()==0:
                    w.writerow(['text','predicted_label','score','correct','timestamp'])
                pred_label = self.last_prediction['label'] if hasattr(self,'last_prediction') else ''
                score = self.last_prediction['probability'] if hasattr(self,'last_prediction') else ''
                w.writerow([self.text_input.get('1.0',tk.END).strip(), pred_label, score, correct, datetime.datetime.utcnow().isoformat()])
            messagebox.showinfo('Feedback','Feedback saved.')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to save feedback: {e}')

if __name__ == '__main__':
    root = tk.Tk()
    App(root)
    root.mainloop()
