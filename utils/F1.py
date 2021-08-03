

def F1_score(TP, FP, FN, TN):
    pre = TP / (TP + FP + 0.001)

    rec = TP / (TP + FN + 0.001)

    acc = (TP + TN) / (TP + FP + TN + FN + 0.001)

    F1 = 2 * (pre * rec) / (pre + rec + 0.001)

    return acc, F1
