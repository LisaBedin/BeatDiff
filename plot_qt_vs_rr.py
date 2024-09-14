import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from functools import partial


def fit_framingham(rrs, qts):
    qtc = (qts + 0.154 * (1 - rrs)).mean()
    pred_fun = lambda x: qtc - (1 - x)*0.154
    return pred_fun, r2_score(y_true=qts, y_pred=pred_fun(rrs))


def fit_root_law(rrs, qts, offset=False, coeff=2):
    lr = LinearRegression(fit_intercept=offset).fit((rrs**(1/coeff))[:, None], qts)
    pred_fun = lambda x: lr.predict(x[:, None]**(1/coeff))
    return pred_fun, r2_score(y_true=qts, y_pred=pred_fun(rrs))


@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> list:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)
    data = np.load(os.path.join(cfg.checkpoint, 'qt_as_rr.npz'))
    qts_mean = data.get('qts_mean')
    qts_interval = data.get('qts_interval')
    rrs = data.get('rrs')[:20]

    goodness_of_fits = {
        "Framingham": [],
        "Bazett": [],
        "Bazett with offset": [],
        "Fridericia": [],
        "Fridericia with offset": [],
    }
    funs = {
        "Framingham": fit_framingham,
        "Bazett": partial(fit_root_law, offset=False, coeff=2),
        "Bazett with offset": partial(fit_root_law, offset=True, coeff=2),
        "Fridericia": partial(fit_root_law, offset=False, coeff=3),
        "Fridericia with offset": partial(fit_root_law, offset=True, coeff=3)
    }
    for patient_qts in qts_mean:
        for method_name, method_fun in funs.items():
            if any(patient_qts != patient_qts):
                goodness_of_fits[method_name].append(-1)
            else:
                goodness_of_fits[method_name].append(method_fun(rrs, patient_qts)[-1])
    goodness_of_fits = {k: np.array(v) for k, v in goodness_of_fits.items()}
    goodness_of_fits_clean = {k: [i for i in v if i != -1] for k, v in goodness_of_fits.items()}
    r2s = [{"method": k, "R2": f'{np.mean(v):.2f} +/- {np.std(v)*1.96 / (len(v)**.5):.2f}'} for k, v in goodness_of_fits_clean.items()]
    print(r2s)
    pd.DataFrame.from_records(r2s).to_csv('data/goodness_of_fits_qt_vs_rr.csv')

    for method_name, goodness_of_fit in goodness_of_fits.items():
        best_three = np.random.randint(low=0, high=len(goodness_of_fit), size=(5,))
        while any(goodness_of_fit[best_three] == -1):
            best_three = np.random.randint(low=0, high=len(goodness_of_fit), size=(5,))#p.argsort(goodness_of_fit)[-3:]
        plt.rcParams.update({'font.size': 18})
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        fig.subplots_adjust(right=1, left=.2, top=1, bottom=.1)
        n_patients = 0
        for patient_qts, patient_error, color in zip(qts_mean[best_three], qts_interval[best_three],
                                                     cm.cool(np.linspace(0., 1, len(best_three)))[::-1]):
            if n_patients > 3:
                break
            pred_fun = funs[method_name](rrs, patient_qts)[0]
            span = np.linspace(min(rrs) * .95, max(rrs) * 1.05, 100)
            curve = pred_fun(span)
            ax.fill_between(x=rrs,
                            y1=patient_qts-patient_error,
                            y2=patient_qts+patient_error,
                            color=color,
                            alpha=.2,
                            interpolate=True)
            # ax.errorbar(x=rrs, y=patient_qts, yerr=patient_error, c=color, capsize=10, fmt="none", alpha=.8)
            ax.plot(span, curve, c=color, alpha=0.5)  #, ls='--')
            ax.scatter(rrs, patient_qts, color=color, alpha=0.5, s=70, marker='.')
            n_patients+=1
        ax.set_xlim(0.6, 1.2) #min(span), max(span))
        # plt.show()
        fig.savefig(f'images/{method_name.replace(" ", "_")}_QT_as_RR.pdf')
        plt.close(fig)
    #goodness_of_fit = [g for g in goodness_of_fit if g != -1]
    #print(f'{np.mean(goodness_of_fit):.2f} ± {np.std(goodness_of_fit)*1.96 / (len(goodness_of_fit)**.5):.2f}')
    print("yes")

if __name__ == '__main__':
    main()