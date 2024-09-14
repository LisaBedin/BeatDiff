import pandas as pd
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from beat_net.beat_net.data_loader import PhysionetECG
matplotlib.rc('font', **{'size'   : 22})


def get_infos(all_feats, set_name, n_beats):
    #all_feats = [ecg_set[k][1].astype(np.float32) for k in tqdm(range(len(ecg_set)))]
    #all_feats = np.stack(all_feats)
    # n_M = int(all_feats[:, 0].sum())
    # n_F = int(all_feats[:, 1].sum())
    ages = all_feats[:, 3]
    all_feats = all_feats[(ages<=1)*(ages>=-1)]
    n_beats = np.array(n_beats)[(ages<=1)*(ages>=-1)]
    #ages[ages>1] = 1
    #ages[ages<0] = 0
    df = pd.DataFrame({'sex': ['M'*int(f_k) + 'F'*int(1-f_k) for f_k in all_feats[:, 0]],
                       'rr': all_feats[:, 2].numpy().astype(np.float32),
                       'age': all_feats[:, 3].numpy().astype(np.float32)*50+50,
                       'set': [set_name]*all_feats.shape[0],
                       'beats': n_beats
                       })
    return df


@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    # cfg = compose(config_name="config") # for loading cfg in python console
    yaml_conf = str(OmegaConf.to_yaml(cfg))

    OmegaConf.set_struct(cfg, False)
    print(yaml_conf)

    train_set = PhysionetECG(database_path=cfg.db_path, categories_to_filter=["NSR", "SB", "STach", "SA"], estimate_noise_std=False,
                             normalized=True, training_class='Training',
                             all=False,
                             return_beat_id=False)
    train_beats = [row.n_beats for row in train_set._ids]
    train_loader = DataLoader(dataset=train_set, batch_size=len(train_set), shuffle=False, num_workers=10)
    for _, train_feats in train_loader:
        break
    df_train = get_infos(train_feats, 'train', train_beats)
    # scipy.stats.sem(train_beats) * scipy.stats.t.ppf(0.95, len(train_beats)-1)
    val_set = PhysionetECG(database_path=cfg.db_path, categories_to_filter=["NSR", "SB", "STach", "SA"], estimate_noise_std=False,
                             normalized=True, training_class='CV',
                             all=False,
                             return_beat_id=False)
    val_beats = [row.n_beats for row in val_set._ids]
    val_loader = DataLoader(dataset=val_set, batch_size=len(val_set), shuffle=False, num_workers=10)
    for _, val_feats in val_loader:
        break
    df_val = get_infos(val_feats, 'val', val_beats)


    test_set = PhysionetECG(database_path=cfg.db_path, categories_to_filter=["NSR", "SB", "STach", "SA"], estimate_noise_std=False,
                             normalized=True, training_class='Test',
                             all=False,
                             return_beat_id=False)
    test_beats = [row.n_beats for row in test_set._ids]
    test_loader = DataLoader(dataset=test_set, batch_size=len(test_set), shuffle=False, num_workers=10)
    for _, test_feats in test_loader:
        break
    df_test = get_infos(test_feats, 'test', test_beats)

    MI_set = PhysionetECG(database_path=cfg.db_path, categories_to_filter=['MI', ], estimate_noise_std=False,
                          normalized=True, training_class='Test',
                          all=True,
                          return_beat_id=False)
    MI_beats = [row.n_beats for row in MI_set._ids]
    MI_loader = DataLoader(dataset=MI_set, batch_size=len(MI_set), shuffle=False, num_workers=10)
    for _, MI_feats in MI_loader:
        break
    df_MI = get_infos(MI_feats, 'MI',MI_beats)
    df_beats = pd.DataFrame({'beats': np.concatenate([train_beats, test_beats, MI_beats]),
                             'split': ['train']*len(train_beats)+['test']*len(test_beats)+['MI']*len(MI_beats)})

    df = pd.concat([df_train, df_val, df_test, df_MI], axis=0)

    dic_color = {'F': '#FF81C0', 'M': 'blue'}
    #dic_color = {'F': 'pink', 'M': 'cyan'}
    left_dic = {'train': 0.17, 'val': 0.13, 'test': 0.13, 'MI': 0.1}
    for set_name in ['train', 'val', 'test', 'MI']:
        fig = plt.figure(figsize=(5,5))
        #.displot(df[df['set'] == set_name],
        #            x='age', hue='sex',
        #            hue_order=['M', 'F'],
        #            palette=dic_color)
        df_tmp = df[df['set'] == set_name]
        plt.hist(df_tmp[df_tmp['sex']=='F']['age'], color=dic_color['F'], edgecolor='k')#, alpha=0.5)
        plt.hist(df_tmp[df_tmp['sex']=='M']['age'], color=dic_color['M'], edgecolor='k', alpha=0.5)
        fig.subplots_adjust(top=1., left=left_dic[set_name], bottom=0.07, right=1)
        fig.savefig(f'images/{set_name}_age_by_sex.pdf')
        print('saved')
        plt.show()
    print('ok')



if __name__ == '__main__':
    main()

    '''
    database_path = '/mnt/data/gabriel/physionet.org/beats_db_more_meta_no_zeros'
    Dx_map = pd.read_csv('Dx_map_physionet.csv')
    label = 'MI-NSR-abQRS'  # 'LAD-LAnFB-NSR'#'LBBB'#'MI-NSR-abQRS'#'LVH-NSR'#'LBBB'#'LVH-NSR'

    categories_healthy = ["NSR", "SB", "STach", "SA"]
    categories_all = Dx_map['Abbreviation'].tolist()

    categories, number_patients, is_healthy = [], [], []
    for cat in categories_all:
        phys_cat = PhysionetECG(database_path=database_path, categories_to_filter=[cat, ], normalized=True,
                                training_class='CV', all=cat not in categories_healthy)
        categories.append(cat)
        number_patients.append(len(phys_cat))
        is_healthy.append(int(cat in categories_healthy))

    df = pd.DataFrame({'Abbreviation': categories, 'Occurences': number_patients, 'IsHealthy': is_healthy})
    df.to_csv('Dx_occ.csv', index=False)
    '''

