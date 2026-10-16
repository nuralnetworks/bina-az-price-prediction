"""One command reproduces every headline number and figure: python -m src.run_all."""
from __future__ import annotations

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import data_prep, evaluate
from .decision_tree import DecisionTree
from .svm import PegasosSVM, rff_map

FIG = os.path.join("report", "figures")
os.makedirs(FIG, exist_ok=True)
os.makedirs("results", exist_ok=True)


def cv5(X, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    folds = np.array_split(idx, 5)
    return [np.concatenate([folds[i] for i in range(5) if i != k]) for k in range(5)], \
           [folds[k] for k in range(5)]


def main() -> None:
    t_all = time.time()
    # toy sanity check: my split search vs sklearn on a tiny fixed example
    from sklearn.tree import DecisionTreeClassifier as _SKT
    _X = np.array([[1.0], [2.0], [3.0], [4.0]])
    _y = np.array([0, 0, 1, 1])
    _m = DecisionTree(task="classification", max_depth=1).fit(_X, _y)
    _s = _SKT(max_depth=1, random_state=42).fit(_X, _y)
    assert abs(_m.root.threshold - _s.tree_.threshold[0]) < 1e-9, "toy split mismatch"
    assert (_m.predict(_X) == _s.predict(_X)).all()
    print(f"toy check ok (split at {_m.root.threshold:g}, predictions match sklearn)")
    df = data_prep.clean(data_prep.load_raw())
    X, y, names = data_prep.make_features(df)
    Xtr, ytr, Xva, yva, Xte, yte = data_prep.train_val_test_split(X, y)
    threshold = float(np.median(ytr))
    ytr_t, _ = data_prep.make_tier_label(ytr, threshold)
    yva_t, _ = data_prep.make_tier_label(yva, threshold)
    yte_t, _ = data_prep.make_tier_label(yte, threshold)
    print(f"rows={len(df)} feats={len(names)} "
          f"train/val/test={len(Xtr)}/{len(Xva)}/{len(Xte)} tier_thr={threshold:.0f}")

    lytr, lyva, lyte = np.log(ytr), np.log(yva), np.log(yte)

    # Task A: depth sweep (log-price) -------------------------------------
    depths = [4, 6, 8, 12, 16, None]
    ra, curve_a = {}, []
    for d in depths:
        m = DecisionTree(task="regression", criterion="mse", max_depth=d).fit(Xtr, lytr)
        r = evaluate.rmse(yva, np.exp(m.predict(Xva)))
        curve_a.append(r)
        ra[d] = m
        print(f"tree-reg depth={d} val_RMSE={r:.0f}")
    best_d_reg = depths[int(np.argmin(curve_a))]
    # one more knob: min_samples_leaf
    leaf_curve = []
    for leaf in [1, 5, 20]:
        m = DecisionTree(task="regression", criterion="mse",
                         max_depth=best_d_reg, min_samples_leaf=leaf).fit(Xtr, lytr)
        r = evaluate.rmse(yva, np.exp(m.predict(Xva)))
        leaf_curve.append(r)
        print(f"tree-reg leaf={leaf} val_RMSE={r:.0f}")
    best_leaf_reg = [1, 5, 20][int(np.argmin(leaf_curve))]

    # Task B: gini vs entropy + depth sweep --------------------------------
    depths_b = [4, 6, 8, 12, 16]
    crit_curve = {}
    for crit in ["gini", "entropy"]:
        f1s = []
        for d in depths_b:
            m = DecisionTree(task="classification", criterion=crit, max_depth=d).fit(Xtr, ytr_t)
            _, _, f1 = evaluate.precision_recall_f1(yva_t, m.predict(Xva))
            f1s.append(f1)
        crit_curve[crit] = f1s
        print(f"tree-clf {crit}: " + " ".join(f"{v:.3f}" for v in f1s))
    bi = int(np.argmax([max(crit_curve["gini"]), max(crit_curve["entropy"])]))
    best_crit = ["gini", "entropy"][bi]
    best_d_clf = depths_b[int(np.argmax(crit_curve[best_crit]))]
    m_leaf = []
    for leaf in [1, 5, 20]:
        m = DecisionTree(task="classification", criterion=best_crit,
                         max_depth=best_d_clf, min_samples_leaf=leaf).fit(Xtr, ytr_t)
        _, _, f1 = evaluate.precision_recall_f1(yva_t, m.predict(Xva))
        m_leaf.append(f1)
        print(f"tree-clf leaf={leaf} val_F1={f1:.3f}")
    best_leaf_clf = [1, 5, 20][int(np.argmax(m_leaf))]

    # SVM: lambda sweep (linear, standardized) ------------------------------
    Xs_tr, Xs_va, Xs_te = data_prep.standardize(Xtr, Xva, Xte)
    svm_curve, svm_models, svm_times = [], {}, {}
    for lam in [1e-5, 1e-4, 1e-3, 1e-2]:
        t0 = time.time()
        s = PegasosSVM(lambda_=lam, n_iters=20_000).fit(Xs_tr, ytr_t)
        tf = time.time() - t0
        _, _, f1 = evaluate.precision_recall_f1(yva_t, s.predict(Xs_va))
        svm_curve.append(f1)
        svm_models[lam] = s
        svm_times[lam] = tf
        print(f"svm lam={lam:g} val_F1={f1:.3f} margin={s.margin_width():.4f} "
              f"sup={s.support_mask(Xs_tr, ytr_t).mean():.2f} ({tf:.1f}s)")
    best_lam = [1e-5, 1e-4, 1e-3, 1e-2][int(np.argmax(svm_curve))]
    svm_lin = svm_models[best_lam]
    # RFF gamma sweep at best lambda
    gammas = [0.005, 0.02, 0.05]
    rff_curve = []
    for g in gammas:
        Ztr, Zva = rff_map(Xs_tr, D=500, gamma=g), rff_map(Xs_va, D=500, gamma=g)
        s = PegasosSVM(lambda_=best_lam, n_iters=60_000).fit(Ztr, ytr_t)
        _, _, f1 = evaluate.precision_recall_f1(yva_t, s.predict(Zva))
        rff_curve.append(f1)
        print(f"svm-rff gamma={g} val_F1={f1:.3f}")
    best_g = gammas[int(np.argmax(rff_curve))]
    Ztr = rff_map(Xs_tr, D=500, gamma=best_g)
    Zte = rff_map(Xs_te, D=500, gamma=best_g)
    svm_rff = PegasosSVM(lambda_=best_lam, n_iters=60_000).fit(Ztr, ytr_t)

    # Finals (val-chosen) + sklearn baselines -------------------------------
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.linear_model import SGDClassifier

    def timed(fit_fn, pred_fn):
        t0 = time.time()
        fit_fn()
        tf = time.time() - t0
        t0 = time.time()
        out = pred_fn()
        return out, tf, time.time() - t0

    tree_reg = DecisionTree(task="regression", criterion="mse",
                            max_depth=best_d_reg, min_samples_leaf=best_leaf_reg)
    tree_clf = DecisionTree(task="classification", criterion=best_crit,
                            max_depth=best_d_clf, min_samples_leaf=best_leaf_clf)
    sk_reg = DecisionTreeRegressor(criterion="squared_error", max_depth=best_d_reg,
                                   min_samples_leaf=best_leaf_reg, random_state=42)
    sk_clf = DecisionTreeClassifier(criterion=best_crit, max_depth=best_d_clf,
                                    min_samples_leaf=best_leaf_clf, random_state=42)
    sk_sgd = SGDClassifier(loss="hinge", alpha=best_lam, max_iter=2000, tol=1e-3, random_state=42)

    pa, tfit, tpred = timed(lambda: tree_reg.fit(Xtr, lytr), lambda: np.exp(tree_reg.predict(Xte)))
    pc, _, _ = timed(lambda: tree_clf.fit(Xtr, ytr_t), lambda: tree_clf.predict(Xte))
    t0 = time.time()
    ps_lin = svm_lin.predict(Xs_te)
    tpred_svm = time.time() - t0
    tfit_svm = svm_times[best_lam]
    t0 = time.time()
    ps_rff = svm_rff.predict(Zte)
    _ = time.time() - t0
    sk_reg.fit(Xtr, lytr)
    sk_clf.fit(Xtr, ytr_t)
    sk_sgd.fit(Xs_tr, np.where(ytr_t == 1, 1, -1))
    pra = np.exp(sk_reg.predict(Xte))
    psc = sk_clf.predict(Xte)
    psg = np.where(sk_sgd.predict(Xs_te) == 1, 1, 0)
    auc_sgd = evaluate.roc_auc(yte_t, sk_sgd.decision_function(Xs_te))

    res = {
        "tree_reg": {"rmse": evaluate.rmse(yte, pa), "mae": evaluate.mae(yte, pa),
                     "r2": evaluate.r2(yte, pa),
                     "log_rmse": evaluate.rmse(lyte, np.log(pa))},
        "sk_reg": {"rmse": evaluate.rmse(yte, pra), "mae": evaluate.mae(yte, pra),
                   "r2": evaluate.r2(yte, pra)},
        "tree_clf": {"f1": evaluate.precision_recall_f1(yte_t, pc)[2],
                     "auc": evaluate.roc_auc(yte_t, tree_clf.predict_proba(Xte)[:, 1])},
        "svm_lin": {"f1": evaluate.precision_recall_f1(yte_t, ps_lin)[2],
                    "auc": evaluate.roc_auc(yte_t, svm_lin.decision_function(Xs_te)),
                    "margin": svm_lin.margin_width(),
                    "sup_frac": float(svm_lin.support_mask(Xs_te, yte_t).mean())},
        "svm_rff": {"f1": evaluate.precision_recall_f1(yte_t, ps_rff)[2],
                    "auc": evaluate.roc_auc(yte_t, svm_rff.decision_function(Zte))},
        "sk_clf": {"f1": evaluate.precision_recall_f1(yte_t, psc)[2],
                   "auc": evaluate.roc_auc(yte_t, sk_clf.predict_proba(Xte)[:, 1])},
        "sk_sgd": {"f1": evaluate.precision_recall_f1(yte_t, psg)[2], "auc": auc_sgd},
    }
    prc, rec, _ = evaluate.precision_recall_f1(yte_t, pc)
    res["tree_clf"].update({"prec": prc, "rec": rec,
                            "cm": evaluate.confusion_matrix(yte_t, pc).tolist()})

    # 5-fold CV on train for headline uncertainty ---------------------------
    tr_idx, te_idx = cv5(Xtr)
    cv = {"tree_clf_f1": [], "svm_lin_f1": [], "sk_clf_f1": []}
    for a, b in zip(tr_idx, te_idx):
        m = DecisionTree(task="classification", criterion=best_crit,
                         max_depth=best_d_clf, min_samples_leaf=best_leaf_clf).fit(Xtr[a], ytr_t[a])
        cv["tree_clf_f1"].append(evaluate.precision_recall_f1(ytr_t[b], m.predict(Xtr[b]))[2])
        mu, sd = Xtr[a].mean(0), Xtr[a].std(0)
        sd[sd == 0] = 1
        s = PegasosSVM(lambda_=best_lam, n_iters=5_000).fit((Xtr[a] - mu) / sd, ytr_t[a])
        cv["svm_lin_f1"].append(evaluate.precision_recall_f1(
            ytr_t[b], s.predict((Xtr[b] - mu) / sd))[2])
        k = DecisionTreeClassifier(criterion=best_crit, max_depth=best_d_clf,
                                   min_samples_leaf=best_leaf_clf, random_state=42).fit(Xtr[a], ytr_t[a])
        cv["sk_clf_f1"].append(evaluate.precision_recall_f1(ytr_t[b], k.predict(Xtr[b]))[2])
    for k, v in cv.items():
        res[k] = {"mean": float(np.mean(v)), "std": float(np.std(v))}

    res["cfg"] = {"depth_reg": best_d_reg, "leaf_reg": best_leaf_reg, "crit": best_crit,
                  "depth_clf": best_d_clf, "leaf_clf": best_leaf_clf, "lambda": best_lam,
                  "gamma": best_g, "threshold": threshold, "n_feats": len(names),
                  "n": len(df), "fit_tree": tfit, "pred_tree": tpred,
                  "fit_svm": tfit_svm, "pred_svm": tpred_svm,
                  "minutes": (time.time() - t_all) / 60}
    with open("results/metrics.json", "w") as f:
        json.dump(res, f, indent=1)

    # Figures ---------------------------------------------------------------
    plt.figure()
    plt.hist(y / 1e3, bins=80)
    plt.xlabel("price (thousand AZN)")
    plt.ylabel("count")
    plt.title("Price distribution (heavy-tailed)")
    plt.savefig(f"{FIG}/price_hist.png", dpi=150, bbox_inches="tight")
    plt.figure()
    plt.hist(np.log(y), bins=80)
    plt.xlabel("log(price)")
    plt.ylabel("count")
    plt.title("log(price) distribution")
    plt.savefig(f"{FIG}/logprice_hist.png", dpi=150, bbox_inches="tight")

    plt.figure()
    xd = [str(d) for d in depths]
    plt.plot(xd, curve_a, marker="o", label="tree-reg val RMSE")
    plt.xlabel("max_depth")
    plt.ylabel("val RMSE (AZN)")
    plt.title("Tree validation curve (Task A)")
    plt.legend()
    plt.savefig(f"{FIG}/tree_val_curve.png", dpi=150, bbox_inches="tight")

    plt.figure()
    plt.plot(depths_b, crit_curve["gini"], marker="o", label="gini")
    plt.plot(depths_b, crit_curve["entropy"], marker="s", label="entropy")
    plt.xlabel("max_depth")
    plt.ylabel("val F1")
    plt.title("Gini vs entropy (Task B)")
    plt.legend()
    plt.savefig(f"{FIG}/tree_criterion.png", dpi=150, bbox_inches="tight")

    plt.figure()
    plt.plot([str(v) for v in [1e-5, 1e-4, 1e-3, 1e-2]], svm_curve, marker="o")
    plt.xlabel("lambda")
    plt.ylabel("val F1")
    plt.title("SVM margin/accuracy trade-off")
    plt.savefig(f"{FIG}/svm_lambda.png", dpi=150, bbox_inches="tight")
    plt.figure()
    plt.plot(svm_lin.objective_)
    plt.xlabel("checkpoint")
    plt.ylabel("J(w, b)")
    plt.title("Pegasos objective (decreasing)")
    plt.savefig(f"{FIG}/svm_objective.png", dpi=150, bbox_inches="tight")

    plt.figure()
    imp = tree_clf.feature_importances_
    top = np.argsort(imp)[-12:][::-1]
    plt.barh([names[i][:22] for i in top][::-1], imp[top][::-1])
    plt.xlabel("impurity decrease share")
    plt.title("Tree feature importances (Task B)")
    plt.savefig(f"{FIG}/importance.png", dpi=150, bbox_inches="tight")

    plt.figure()
    cm = np.array(res["tree_clf"]["cm"])
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xticks([0, 1], ["standard", "premium"])
    plt.yticks([0, 1], ["standard", "premium"])
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.title("Confusion matrix (my tree, Task B)")
    plt.savefig(f"{FIG}/confusion.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # Console table ----------------------------------------------------------
    print(f"\nFINAL depth_reg={best_d_reg} leaf_reg={best_leaf_reg} "
          f"crit={best_crit} depth_clf={best_d_clf} leaf_clf={best_leaf_clf} "
          f"lambda={best_lam:g} gamma={best_g}")
    evaluate.compare({
        "my tree-reg": {"RMSE": res["tree_reg"]["rmse"], "MAE": res["tree_reg"]["mae"],
                          "R2": res["tree_reg"]["r2"]},
        "sklearn tree-reg": {"RMSE": res["sk_reg"]["rmse"], "MAE": res["sk_reg"]["mae"],
                             "R2": res["sk_reg"]["r2"]},
        "my tree-clf": {"F1": res["tree_clf"]["f1"], "AUC": res["tree_clf"]["auc"]},
        "my svm-lin": {"F1": res["svm_lin"]["f1"], "AUC": res["svm_lin"]["auc"]},
        "my svm-rff": {"F1": res["svm_rff"]["f1"], "AUC": res["svm_rff"]["auc"]},
        "sklearn tree-clf": {"F1": res["sk_clf"]["f1"], "AUC": res["sk_clf"]["auc"]},
        "sklearn sgd-hinge": {"F1": res["sk_sgd"]["f1"], "AUC": res["sk_sgd"]["auc"]},
    })
    for k, v in cv.items():
        print(f"{k}: {np.mean(v):.3f} +- {np.std(v):.3f}")
    print(f"done in {res['cfg']['minutes']:.1f} min")


if __name__ == "__main__":
    main()
