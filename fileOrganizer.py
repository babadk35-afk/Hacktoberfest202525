"""
Auto File Organizer with ML Labeler
- Learns from existing folders (e.g., 'images','docs','data') by reading file names and extensions
- Trains a classifier to predict target folder for new files and moves them
- Plots distribution per predicted class
Usage:
  python 20_file_organizer_ml.py watch_folder images docs data
Dependencies: scikit-learn, matplotlib
"""
import sys, os, shutil, re, matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def build_training(paths):
    X, y = [], []
    for label, folder in enumerate(paths):
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                X.append(name)
                y.append(label)
    return X, y

if __name__ == "__main__":
    if len(sys.argv) < 3: print(__doc__); sys.exit(0)
    watch = sys.argv[1]; targets = sys.argv[2:]
    X, y = build_training(targets)
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2,5)).fit(X)
    Xv = vec.transform(X)
    clf = LogisticRegression(max_iter=200).fit(Xv, y)
    preds = []
    for name in os.listdir(watch):
        path = os.path.join(watch, name)
        if not os.path.isfile(path): continue
        p = clf.predict(vec.transform([name]))[0]
        dest = os.path.join(targets[p], name)
        shutil.move(path, dest)
        preds.append(p)
        print(f"Moved {name} -> {targets[p]}")
    if preds:
        counts = [preds.count(i) for i in range(len(targets))]
        plt.figure(); plt.bar([os.path.basename(t) for t in targets], counts); plt.title("Moved Files by Predicted Class"); plt.show()
