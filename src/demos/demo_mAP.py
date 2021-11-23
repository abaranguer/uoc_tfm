import numpy as np
from sklearn.metrics import average_precision_score

y_labels = np.array([0, 0, 1, 1, 1, 2, 2])

y_scores = np.array([
    [0.1, 0.4, 0.35, 0.8, 0.2, 0.3, 0.6],  # prediccion de clasificador 1
    [0.2, 0.3, 0.4, 0.1, 0.5, 0.2, 0.3],  # prediccion de clasificador 2
    [0.7, 0.3, 0.25, 0.1, 0.3, 0.5, 0.1],  # prediccion de clasificador 3
])

# la suma de las columnas y_scores da 1

print("La suma de las columnas de y_scores da 1")
print(y_scores.sum(0))

# ============= map por clase ==============
mAPs = []
for cls_id in range(3):
    y_true = np.atleast_2d(y_labels == cls_id).T
    y_score = np.atleast_2d(y_scores[cls_id, :]).T
    mAP = average_precision_score(y_true, y_score)
    mAPs.append(mAP)
    print(f"clase {cls_id} mAP: {mAP}")

# mAP ponderada
freqs = np.array([sum(y_labels == x) for x in range(3)]) / y_labels.size
ws = sum(freqs) / freqs
ws = ws / sum(ws)

for fr, w, cls_id in zip(freqs, ws, range(3)):
    print(f"clase {cls_id} ocupa {fr * 100:.4}% en la base de datos --> su peso: {w:.4}")

mAP = ws @ mAPs
print(f"mAP ponderada: {mAP}")
print(f"mAP promedio: {np.mean(mAPs)}")
