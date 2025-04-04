# ------------------------------------------------------------ #
#                     _ _  __     
#    _ _ __ _ _ _  __| (_)/ _|___ 
#   | '_/ _` | ' \/ _` | |  _/ -_)
#   |_| \__,_|_||_\__,_|_|_| \___|
#------------------------------------
#Future Predictor (w/ Random Simulating)
#------------------------------------
#
#====================================
#              LINKS
#  -------------------------------
#
#+ GitHub: https://randife.com/github
#
#+ Kaggle: https://randife.com/kaggle
#
#
#====================================
#              ABOUT
#  -------------------------------
#
#Life includes both random events and rule based events.
#If we have enough data about past events to find about
#rules in them, by simulating random events and finding
#optimized event pairs (based on found rules), we can
#predict future.
#
#Randife is a framework for predicting future with
#random simulating. It includes some sub projects
#which implement Randife for predicting specified
#events, such as:
#
#+ Orottick4RL:
#  o Website: https://orottick4.randife.com
#  o About: Predict Oregon Lottery - Pick 4
#
#+ PwrallRL:
#  o Website: https://pwrall.randife.com
#  o About: Predict Power Ball
#
#
#====================================
#            Copyright
#  -------------------------------
#
# Randife is written and copyrighted by Dinh Thoai Tran
# <dinhtt@randrise.com> [https://dinhtt.randrise.com]
#
#
#====================================
#            License
#  -------------------------------
#
# Randife is distributed under Apache-2.0 license
# [ https://github.com/dinhtt-randrise/randife/blob/main/LICENSE ]
#
# ------------------------------------------------------------ #

import random
import os
import json
import time
from datetime import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
import pickle
import glob

class RandifeRandomGenerator:
    def __init__(self, rnd_format):
        self.rnd_format = rnd_format

    def set_seed(self, sim_seed):
        random.seed(sim_seed)

    def randint(self, n_min, n_max):
        return random.randint(n_min, n_max)

    def rnd_kind(self):
        return 'rnd_sng_thd'
        
class RandifeRandomFormat:
    def __init__(self, size = 1, rnd_min_list = [0], rnd_max_list = [9999], def_rnd_min = 0, def_rnd_max = 9999, err_rnd_num = -1, load_cache_dir = '/kaggle/working', save_cache_dir = '/kaggle/working'):
        if len(rnd_min_list) < size:
            size = len(rnd_min_list)
        if len(rnd_max_list) < size:
            size = len(rnd_max_list)
        self.size = size
        self.rnd_min_list = rnd_min_list
        self.rnd_max_list = rnd_max_list
        self.def_rnd_min = def_rnd_min
        self.def_rnd_max = def_rnd_max
        self.err_rnd_num = err_rnd_num

        self.cache_capture_seed = {}
        self.cache_reproduce = {}
        self.cache_capture = {}

        os.system(f'mkdir -p "{load_cache_dir}"')
        self.load_cache_dir = load_cache_dir

        os.system(f'mkdir -p "{save_cache_dir}"')
        self.save_cache_dir = save_cache_dir

        self.load_cache()

        self.rnd_gen = self.create_random_generator()

        self.dict_time_data = {}
        
    #----- Create functions -----#
    
    def create_random_generator(self):
        return RandifeRandomGenerator(self)

    def create_random_simulator(self):
        return RandifeRandomSimulator(self)

    #----- Cache functions -----#

    def save_cache(self):
        cdir = self.save_cache_dir

        fn = f'{cdir}/rl_cache_capture_seed.pkl'
        with open(fn, 'wb') as f:
            pickle.dump(self.cache_capture_seed, f)
            print('=> [cache_capture_seed] Saved')

        fn = f'{cdir}/rl_cache_reproduce.pkl'
        with open(fn, 'wb') as f:
            pickle.dump(self.cache_reproduce, f)
            print('=> [cache_reproduce] Saved')

        fn = f'{cdir}/rl_cache_capture.pkl'
        with open(fn, 'wb') as f:
            pickle.dump(self.cache_capture, f)
            print('=> [cache_capture] Saved')

    def load_cache(self):
        cdir = self.load_cache_dir
        
        fn = f'{cdir}/rl_cache_capture_seed.pkl'
        if os.path.exists(fn):
            with open(fn, 'rb') as f:
                self.cache_capture_seed = pickle.load(f)
                print('=> [cache_capture_seed] Loaded')

        fn = f'{cdir}/rl_cache_reproduce.pkl'
        if os.path.exists(fn):
            with open(fn, 'rb') as f:
                self.cache_reproduce = pickle.load(f)
                print('=> [cache_reproduce] Loaded')

        fn = f'{cdir}/rl_cache_capture.pkl'
        if os.path.exists(fn):
            with open(fn, 'rb') as f:
                self.cache_capture = pickle.load(f)
                print('=> [cache_capture] Loaded')

    #----- Random number functions -----#

    def set_seed(self, sim_seed):
        self.rnd_gen.set_seed(sim_seed)
        
    def rnd_num(self, mix = 0):
        vmin, vmax = self.get_min_max(mix)
        return self.rnd_gen.randint(vmin, vmax)

    def rnd_num_list(self):
        num_list = []
        for mix in range(self.size):
            n = self.rnd_num(mix)
            num_list.append(n)
        return num_list

    def reproduce(self, sim_seed, sim_cnt = 1):
        if sim_cnt <= 0:
            return self.get_err_rnd_num_list()

        key = f'{sim_seed}_{sim_cnt}'
        if key in self.cache_reproduce:
            return self.cache_reproduce[key]
            
        self.set_seed(sim_seed)
        rnd_num_list = self.get_err_rnd_num_list()
        for ci in range(sim_cnt):
            rnd_num_list = self.rnd_num_list()

        self.cache_reproduce[key] = rnd_num_list
        
        return rnd_num_list

    def capture_seed(self, sim_cnt, rnd_num_list):
        ls_rnd_num = [str(x) for x in rnd_num_list]
        key = f'{sim_cnt}_' + '_'.join(ls_rnd_num)
        if key in self.cache_capture_seed:
            return self.cache_capture_seed[key]
            
        sim_seed = 0
        prd_rnd_num_list = self.reproduce(sim_seed, sim_cnt)
            
        while not self.match(rnd_num_list, prd_rnd_num_list, 'ma'):
            sim_seed += 1
            prd_rnd_num_list = self.reproduce(sim_seed, sim_cnt)

        self.cache_capture_seed[key] = sim_seed
        
        return sim_seed

    def capture(self, win_rnd_num_list, prd_rnd_num_list):
        lx_win_rnd_num = [str(x) for x in win_rnd_num_list]
        lx_prd_rnd_num = [str(x) for x in prd_rnd_num_list]
        key = '_'.join(lx_win_rnd_num) + '__' + '_'.join(lx_prd_rnd_num)
        if key in self.cache_capture:
            return self.cache_capture[key][0], self.cache_capture[key][1]
                        
        sim_seed = self.capture_seed(1, prd_rnd_num_list)

        self.set_seed(sim_seed)
        
        sim_cnt = 0
        p = self.rnd_num_list()
        sim_cnt += 1

        while not self.match(win_rnd_num_list, p, 'ma'):
            p = self.rnd_num_list()
            sim_cnt += 1

        pn = self.reproduce(sim_seed, 1)
        pw = self.reproduce(sim_seed, sim_cnt)
        
        if self.match(pw, win_rnd_num_list, 'ma') and self.match(pn, prd_rnd_num_list, 'ma'):
            self.cache_capture[key] = [sim_seed, sim_cnt]
            return sim_seed, sim_cnt
        else:
            self.cache_capture[key] = [-1, -1]
            return -1, -1

    #----- Log functions -----#

    def heading(self, method, kind):
        if method == 'simulate':
            if kind == 'method_start':
                text = '''
====================================
        ANALYZE SIMULATION
  -------------------------------
                '''
                return text
            if kind == 'method_end':
                text = '''
  -------------------------------
        ANALYZE SIMULATION
====================================
                '''
                return text
            if kind == 'parameters_start':
                text = '''
  -------------------------------
            PARAMETERS
  -------------------------------
                '''
                return text
            if kind == 'parameters_end':
                text = '''
  -------------------------------
                '''
                return text
            if kind == 'prediction_start':
                text = '''
  -------------------------------
           PREDICTION
  -------------------------------
                '''
                return text
            if kind == 'prediction_end':
                text = '''
  -------------------------------
                '''
                return text

        if method == 'map_collect':
            if kind == 'method_start':
                text = '''
====================================
           MAP COLLECT
  -------------------------------
                '''
                return text
            if kind == 'method_end':
                text = '''
  -------------------------------
           MAP COLLECT
====================================
                '''
                return text
            if kind == 'parameters_start':
                text = '''
  -------------------------------
            PARAMETERS
  -------------------------------
                '''
                return text
            if kind == 'parameters_end':
                text = '''
  -------------------------------
                '''
                return text

        return ''

    #----- Properties functions -----#

    def get_err_rnd_num(self):
        return self.err_rnd_num

    def get_err_rnd_num_list(self):
        return [self.err_rnd_num for mix in range(self.size)]
        
    def get_def_rnd_min(self):
        return self.def_rnd_min

    def get_def_rnd_max(self):
        return self.def_rnd_max

    def get_size(self):
        return self.size

    def get_min_max(self, mix = 0):
        if mix < 0 or mix >= self.size:
            return self.def_rnd_min, self.def_rnd_max
        return self.rnd_min_list[mix], self.rnd_max_list[mix]

    def has_err_rnd_num(self, rnd_num_list):
        if self.err_rnd_num in rnd_num_list:
            return True
        else:
            return False

    #----- Data functions -----#

    def match(self, wl, nl, match_kind = 'ma'):
        if len(wl) != self.size or len(nl) != self.size:
            return False
        if match_kind == 'ma':
            wls = [str(x) for x in wl]
            nls = [str(x) for x in nl]
            ws = '_'.join(wls)
            ns = '_'.join(nls)
            if ws == ns:
                return True
            else:
                return False
        else:
            return False

    def reset_dict_time_data(self):
        self.dict_time_data = {}

    def get_column_index(self, ddf, column_name):
        columns = ddf.columns
        for cix in range(len(columns)):
            if columns[cix] == column_name:
                return cix
        return -1

    def export_dict_num(self, var_dict, var_name, var_val):
        var_dict[var_name] = var_val

    def export_dict_num_list(self, var_dict, var_name, nl):
        for mix in range(self.size):
            no = mix + 1
            n = nl[mix]
            self.export_dict_num(var_dict, f'{var_name}_{no}', n)
        
    def export_dataset_num(self, ddf, rwi, var_name, var_val):
        cix = self.get_column_index(ddf, var_name)
        if cix < 0:
            return
        ddf.iloc[rwi, cix] = var_val

    def export_dataset_num_list(self, ddf, rwi, var_name, nl):
        for mix in range(self.size):
            no = mix + 1
            n = nl[mix]
            self.export_dataset_num(ddf, rwi, f'{var_name}_{no}', n)

    def import_dataset_num_list(self, ddf, rwi, var_name):
        nl = []
        for mix in range(self.size):
            no = mix + 1
            n = ddf[f'{var_name}_{no}'].iloc[rwi]
            nl.append(n)
        return nl

    def export_dict_num_list(self, var_dict, nl, var_name):
        for mix in range(self.size):
            no = mix + 1
            var_dict[f'{var_name}_{no}'] = nl[mix]
        
    def import_dict_time_data(self, time_no, data):
        self.dict_time_data[time_no] = data
        
    def export_dict_time_data(self, time_no):
        if time_no not in self.dict_time_data:
            return None
        return self.dict_time_data[time_no]

    def trim_dataset_sim_input(self, ddf):
        cols = ['time_no']
        for mix in range(self.size):
            no = mix + 1
            cols.append(f'w_{no}')
        for mix in range(self.size):
            no = mix + 1
            cols.append(f'n_{no}')
        return ddf[cols]

    def get_pair_matching_keys(self):
        return ['ma']

    def refine_json_pred(self, xdf, json_pred):
        return json_pred

    def capture_map(self, pdf, x_sim_seed):
        return []

    def is_observe_good(self, odf, o_cnt, o_ma_field = 'ma'):
        if len(odf) == o_cnt:
            df = odf[odf[o_ma_field] > 0]
            if len(df) > 0:
                return True
            return False
        else:
            return False   

    def is_pick_good(self, odf, p_cnt):
        if len(odf) == p_cnt:
            return True
        else:
            return False

    def copy_file(self, src_fn, tag_fn):
        os.system(f'cp -f "{src_fn}" "{tag_fn}"')
        
class RandifeRandomSimulator:
    def __init__(self, rnd_format):
        self.rnd_format = rnd_format

    def map_collect(self, o_cnt, p_cnt, ma_field, data_dirs, save_dir, fn_observe_glob, fn_observe_file, fn_pick_file, fn_pred_file):
        text = self.rnd_format.heading('map_collect', 'method_start')
        print(text)

        text = self.rnd_format.heading('map_collect', 'parameters_start')
        print(text)

        print(f'O_CNT: {o_cnt}')
        print(f'P_CNT: {p_cnt}')
        print(f'DATA_DIRS: {data_dirs}')
        print(f'SAVE_DIR: {save_dir}')

        text = self.rnd_format.heading('map_collect', 'parameters_end')
        print(text)
    
        for data_dir in data_dirs:
            obs_glob = fn_observe_glob()
            lg_obs = glob.glob(f'{data_dir}/{obs_glob}')
            for fn_obs in lg_obs:
                odf = pd.read_csv(fn_obs)
                if not self.rnd_format.is_observe_good(odf, o_cnt, ma_field):
                    continue
                observe_fn = fn_observe_file(fn_obs)
                self.rnd_format.copy_file(fn_obs, f'{save_dir}/{observe_fn}')
                print(f'== [Copy] ==> {observe_fn}')
                for ri in range(len(odf)):
                    if odf[ma_field].iloc[ri] > 0:
                        pick_fn = fn_pick_file(odf, ri)
                        pred_fn = fn_pred_file(odf, ri)
                        pdf = pd.read_csv(f'{data_dir}/{pick_fn}')
                        if not self.rnd_format.is_pick_good(pdf, p_cnt):
                            continue
                        self.rnd_format.copy_file(f'{data_dir}/{pick_fn}', f'{save_dir}/{pick_fn}')
                        print(f'== [Copy] ==> {pick_fn}')
                        self.rnd_format.copy_file(f'{data_dir}/{pred_fn}', f'{save_dir}/{pred_fn}')
                        print(f'== [Copy] ==> {pred_fn}')
        
        text = self.rnd_format.heading('map_collect', 'method_end')
        print(text)

    def simulate(self, data_df, prd_time_no, prc_time_cnt, prc_runtime, tck_cnt, map_cnt, has_step_log, cache_only):
        start_time = time.time()
        
        text = self.rnd_format.heading('simulate', 'method_start')
        print(text)

        text = self.rnd_format.heading('simulate', 'parameters_start')
        print(text)

        print(f'PRD_TIME_NO: {prd_time_no}')
        print(f'----------')
        time_data = self.rnd_format.export_dict_time_data(prd_time_no)
        for key in time_data.keys():
            val = time_data[key]
            ukey = key.upper()
            print(f'{ukey}: {val}')
        print(f'----------')
        print(f'PRC_TIME_CNT: {prc_time_cnt}')
        print(f'TCK_CNT: {tck_cnt}')
        print(f'PRC_RUNTIME: {prc_runtime}')
        rs = 'yes' if has_step_log else 'no'
        print(f'HAS_STEP_LOG: {rs}')
        rs = 'yes' if cache_only else 'no'
        print(f'CACHE_ONLY: {rs}')

        text = self.rnd_format.heading('simulate', 'parameters_end')
        print(text)

        data_df = self.rnd_format.trim_dataset_sim_input(data_df)
        
        cix = self.rnd_format.get_column_index(data_df, 'sim_seed')
        if cix < 0:
            data_df['sim_seed'] = -1
        cix = self.rnd_format.get_column_index(data_df, 'sim_cnt')
        if cix < 0:
            data_df['sim_cnt'] = -1

        adf = data_df[data_df['time_no'] <= prd_time_no]
        if len(adf) < 1:
            print(f'== [Error] ==> Dataset is empty!')
            return None, None, None, None, None
            
        xdf = data_df[data_df['time_no'] == prd_time_no]
        if len(xdf) == 0:
            print(f'== [Error] ==> Predicting moment is not found!')
            return None, None, None, None, None

        x_time_no = xdf['time_no'].iloc[0]
        x_nl = self.rnd_format.import_dataset_num_list(xdf, 0, 'n')
        x_wl = self.rnd_format.import_dataset_num_list(xdf, 0, 'w')
        x_sim_seed = self.rnd_format.capture_seed(1, x_nl)
        self.rnd_format.export_dataset_num(xdf, 0, 'sim_seed', x_sim_seed)

        sdf = data_df[data_df['time_no'] < prd_time_no]
        if len(sdf) == 0:
            print(f'== [Error] ==> Dataset is empty!')
            return None, None, None, None, None

        sdf = sdf[(sdf['n_1'] != self.rnd_format.get_err_rnd_num())&(sdf['w_1'] != self.rnd_format.get_err_rnd_num())]
        prc_cnt = prc_time_cnt * 2
        if len(sdf) < prc_cnt:
            sz = len(sdf)
            print(f'== [Error] ==> Dataset does not contain enough data. Its size is {sz} instead of {prc_cnt}.')
            return None, None, None, None, None

        sdf = sdf.sort_values(by=['time_no'], ascending=[False])
        if len(sdf) > prc_cnt:
            sdf = sdf[:prc_cnt]

        sz = len(sdf)
        dbcnt = int(round(sz / 100.0))
        if dbcnt < 1:
            dbcnt = 1
        for rwi in range(sz):
            if time.time() - start_time > prc_runtime:
                break

            nl = self.rnd_format.import_dataset_num_list(sdf, rwi, 'n')
            wl = self.rnd_format.import_dataset_num_list(sdf, rwi, 'w')

            if self.rnd_format.has_err_rnd_num(nl) or self.rnd_format.has_err_rnd_num(wl):
                continue

            sim_seed, sim_cnt = self.rnd_format.capture(wl, nl)
            self.rnd_format.export_dataset_num(sdf, rwi, 'sim_seed', sim_seed)
            self.rnd_format.export_dataset_num(sdf, rwi, 'sim_cnt', sim_cnt)

            pix = rwi + 1
            if pix % dbcnt == 0:
                if has_step_log:
                    print(f'== [S1] ==> {pix} / {sz}')

        try:
            self.rnd_format.save_cache()
        except Exception as e:
            msg = str(e)
            print(f'== [Error] ==> {msg}')

        sdf = sdf.sort_values(by=['time_no'], ascending=[False])

        mdf = sdf.sort_values(by=['time_no'], ascending=[True])

        sz = prc_time_cnt * prc_time_cnt
        dbcnt = int(round(sz / 100.0))
        if dbcnt < 1:
            dbcnt = 1
        dbix = 0
        pix = 0

        if not cache_only:
            data_bag = {}
            for match_kind in self.rnd_format.get_pair_matching_keys():
                data_bag[f'{match_kind}_cnt'] = 0
                mdf[f'{match_kind}'] = 0
                mdf[f'a_{match_kind}'] = 0
                mdf[f'{match_kind}_cnt'] = 0
                mdf[f'p_time_no_{match_kind}'] = ''
                mdf[f'p_sim_seed_{match_kind}'] = ''
                mdf[f'p_win_num_{match_kind}'] = ''
                mdf[f'p_prd_num_{match_kind}'] = ''
            for mix in range(self.rnd_format.get_size()):
                no = mix + 1
                mdf[f'p_{no}'] = self.rnd_format.get_err_rnd_num()
        for pia in range(len(mdf)):
            if time.time() - start_time > prc_runtime:
                break

            if pia <= prc_time_cnt:
                continue

            y_sim_seed = mdf['sim_seed'].iloc[pia]
                            
            if not cache_only:
                y_time_no = mdf['time_no'].iloc[pia]
                y_sim_cnt = mdf['sim_cnt'].iloc[pia]
                y_nl = self.rnd_format.import_dataset_num_list(mdf, pia, 'n')
                y_wl = self.rnd_format.import_dataset_num_list(mdf, pia, 'w')
                if self.rnd_format.has_err_rnd_num(y_nl) or self.rnd_format.has_err_rnd_num(y_wl):
                    continue

                data_bag_a = {}
                for match_kind in self.rnd_format.get_pair_matching_keys():
                    data_bag_a[f'{match_kind}'] = 0
                    data_bag_a[f'a_{match_kind}'] = 0
                    data_bag_a[f'{match_kind}_cnt'] = 0
                    data_bag_a[f'p_time_no_{match_kind}'] = ''
                    data_bag_a[f'p_sim_seed_{match_kind}'] = ''
                    data_bag_a[f'p_win_num_{match_kind}'] = ''
                    data_bag_a[f'p_prd_num_{match_kind}'] = ''
            
            for pib in range(len(mdf)):
                if time.time() - start_time > prc_runtime:
                    break

                if pia == pib:
                    continue
                    
                if pib >= pia:
                    break

                if pib < pia - prc_time_cnt:
                    continue
                    
                pix += 1
                if pix % dbcnt == 0:
                    dbix += 1
                    if has_step_log:
                        print(f'== [S2] ==> {dbix}, {pix} / {sz}')

                z_sim_cnt = sdf['sim_cnt'].iloc[pib]

                if not cache_only:
                    z_time_no = mdf['time_no'].iloc[pib]
                    z_sim_seed = mdf['sim_seed'].iloc[pib]
                    z_nl = self.rnd_format.import_dataset_num_list(mdf, pib, 'n')
                    z_wl = self.rnd_format.import_dataset_num_list(mdf, pib, 'w')
                    if self.rnd_format.has_err_rnd_num(z_nl) or self.rnd_format.has_err_rnd_num(z_wl):
                        continue

                    data_bag_b = {}
                    for match_kind in self.rnd_format.get_pair_matching_keys():
                        data_bag_b[f'i_{match_kind}'] = mdf[f'{match_kind}'].iloc[pib]
                        data_bag_b[f'pi_time_no_{match_kind}'] = str(mdf[f'p_time_no_{match_kind}'].iloc[pib])
                        data_bag_b[f'pi_sim_seed_{match_kind}'] = str(mdf[f'p_sim_seed_{match_kind}'].iloc[pib])
                        data_bag_b[f'pi_win_num_{match_kind}'] = str(mdf[f'p_win_num_{match_kind}'].iloc[pib])
                        data_bag_b[f'pi_prd_num_{match_kind}'] = str(mdf[f'p_prd_num_{match_kind}'].iloc[pib])

                pl = self.rnd_format.reproduce(y_sim_seed, z_sim_cnt)

                if not cache_only:
                    self.rnd_format.export_dataset_num_list(mdf, pia, 'p', pl)

                    for match_kind in self.rnd_format.get_pair_matching_keys():
                        v = 0
                        if self.rnd_format.match(y_wl, pl, match_kind):
                            v = 1

                        data_bag_b[f'i_{match_kind}'] += v
                        data_bag_a[f'a_{match_kind}'] += v

                        if data_bag_b[f'pi_time_no_{match_kind}'] != '':
                            data_bag_b[f'pi_time_no_{match_kind}'] += '; '
                            data_bag_b[f'pi_sim_seed_{match_kind}'] += '; '
                            data_bag_b[f'pi_win_num_{match_kind}'] += '; '
                            data_bag_b[f'pi_prd_num_{match_kind}'] += '; '

                        data_bag_b[f'pi_time_no_{match_kind}'] += str(y_time_no)
                        data_bag_b[f'pi_sim_seed_{match_kind}'] += str(y_sim_seed)
                        data_bag_b[f'pi_win_num_{match_kind}'] += ', '.join([str(x) for x in y_wl])
                        data_bag_b[f'pi_prd_num_{match_kind}'] += ', '.join([str(x) for x in pl])
                        
                        self.rnd_format.export_dataset_num(mdf, pib, f'{match_kind}', data_bag_b[f'i_{match_kind}'])
                        self.rnd_format.export_dataset_num(mdf, pib, f'p_time_no_{match_kind}', data_bag_b[f'pi_time_no_{match_kind}'])
                        self.rnd_format.export_dataset_num(mdf, pib, f'p_sim_seed_{match_kind}', data_bag_b[f'pi_sim_seed_{match_kind}'])
                        self.rnd_format.export_dataset_num(mdf, pib, f'p_win_num_{match_kind}', data_bag_b[f'pi_win_num_{match_kind}'])
                        self.rnd_format.export_dataset_num(mdf, pib, f'p_prd_num_{match_kind}', data_bag_b[f'pi_prd_num_{match_kind}'])
                else:
                    pl2 = self.rnd_format.reproduce(x_sim_seed, z_sim_cnt)
                    
            if not cache_only:
                for match_kind in self.rnd_format.get_pair_matching_keys():
                    data_bag[f'{match_kind}_cnt'] += data_bag_a[f'a_{match_kind}']
                    self.rnd_format.export_dataset_num(mdf, pia, f'{match_kind}_cnt', data_bag[f'{match_kind}_cnt'])

        try:
            self.rnd_format.save_cache()
        except Exception as e:
            msg = str(e)
            print(f'== [Error] ==> {msg}')

        if cache_only:      
            pdf = sdf.sort_values(by=['time_no'], ascending=[False])
            if len(pdf) > prc_time_cnt:
                pdf = pdf[:prc_time_cnt]
            
            for rwi in range(len(pdf)):
                z_sim_cnt = pdf['sim_cnt'].iloc[rwi]
                pl = self.rnd_format.reproduce(x_sim_seed, z_sim_cnt)

            try:
                self.rnd_format.save_cache()
            except Exception as e:
                msg = str(e)
                print(f'== [Error] ==> {msg}')
            
            return None, None, None, None, None

        sdf = pd.concat([xdf, sdf])
        sdf = sdf.sort_values(by=['time_no'], ascending=[False])
        pdf = mdf.sort_values(by=['time_no'], ascending=[False])
        if len(pdf) > prc_time_cnt:
            pdf = pdf[:prc_time_cnt]
        for mix in range(self.rnd_format.get_size()):
            no = mix + 1
            pdf[f'fp_{no}'] = self.rnd_format.get_err_rnd_num()

        ls_pred = []
        ls_sim_cnt = []
        ma_rsi = -1
        for rwi in range(len(pdf)):
            z_sim_cnt = pdf['sim_cnt'].iloc[rwi]
            pl = self.rnd_format.reproduce(x_sim_seed, z_sim_cnt)
            self.rnd_format.export_dataset_num_list(pdf, rwi, 'fp', pl)
            spl = [str(x) for x in pl]
            sp = ', '.join(spl)
            ls_pred.append(sp)
            ls_sim_cnt.append(str(z_sim_cnt))
            if ma_rsi < 0:
                if self.rnd_format.match(x_wl, pl, 'ma'):
                    ma_rsi = rwi
            if tck_cnt > 0:
                if rwi >= tck_cnt:
                    break
                    
        try:
            self.rnd_format.save_cache()
        except Exception as e:
            msg = str(e)
            print(f'== [Error] ==> {msg}')

        mapc = 0
        ma_pred = ''
        ll_ma_pred = self.rnd_format.capture_map(pdf, x_sim_seed)
        if len(ll_ma_pred) > 0:
            mapc = 1
            ls_ma_pred = [', '.join([str(y) for y in x]) for x in ll_ma_pred]
            if map_cnt > 0:
                if len(ls_ma_pred) > map_cnt:
                    ls_ma_pred = ls_ma_pred[:map_cnt]
            ma_pred = '; '.join(ls_ma_pred)
            
        xs_pred = '; '.join(ls_pred)
        xs_sim_cnt = '; '.join(ls_sim_cnt)

        w_ls = [str(x) for x in x_wl]
        n_ls = [str(x) for x in x_nl]
        x_w = ', '.join(w_ls)
        x_n = ', '.join(n_ls)
        json_pred = {'time_no': int(x_time_no), 'w': x_w, 'n': x_n, 'sim_seed': int(x_sim_seed), 'prc_time_cnt': int(prc_time_cnt), 'tck_cnt': int(tck_cnt), 'sim_cnt': xs_sim_cnt, 'pred': xs_pred, 'ma_rsi': int(ma_rsi), 'mapc': mapc, 'ma_pred': ma_pred}
        n_json_pred = self.rnd_format.refine_json_pred(xdf, json_pred)
        
        text = self.rnd_format.heading('simulate', 'prediction_start')
        print(text)

        print(str(json_pred))
        
        text = self.rnd_format.heading('simulate', 'prediction_end')
        print(text)

        print(str(n_json_pred))
        
        text = self.rnd_format.heading('simulate', 'prediction_end')
        print(text)

        text = self.rnd_format.heading('simulate', 'method_end')
        print(text)

        return sdf, mdf, pdf, json_pred, n_json_pred
