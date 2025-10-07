from lime.lime_text import LimeTextExplainer
from typing import List, Callable
import numpy as np


def lime_explain(predict_proba_fn: Callable[[List[str]], np.ndarray], class_names, text: str, num_features=8):
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(text, predict_proba_fn, num_features=num_features)
    return exp
