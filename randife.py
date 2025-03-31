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
#              _   _   _    _   _ _  
#  ___ _ _ ___| |_| |_(_)__| |_| | | 
# / _ \ '_/ _ \  _|  _| / _| / /_  _|
# \___/_| \___/\__|\__|_\__|_\_\ |_| 
#------------------------------------
# Oregon Lottery - Pick 4 Predictor
#------------------------------------
#
#====================================
#            LINKS
#  -------------------------------
#
# + Kaggle: https://orottick4.com/kaggle
#
# + GitHub: https://orottick4.com/github
#
# + Lottery: https://orottick4.com/lotte
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

class RandifeRandomFormat:
    def __init__(self, moment_size = 1, rnd_min_list = [0], rnd_max_list = [9999], def_rnd_min = 0, def_rnd_max = 9999, err_rnd_num = -1, load_cache_dir = '/kaggle/working', save_cache_dir = '/kaggle/working', fn_rnd_num_before = None, fn_rnd_num_after = None, fn_rnd_num_list_before = None, fn_rnd_num_list_after = None):
        if len(rnd_min_list) < moment_size:
            moment_size = len(rnd_min_list)
        if len(rnd_max_list) < moment_size:
            moment_size = len(rnd_max_list)
        self.moment_size = moment_size
        self.rnd_min_list = rnd_min_list
        self.rnd_max_list = rnd_max_list
        self.def_rnd_min = def_rnd_min
        self.def_rnd_max = def_rnd_max
        self.err_rnd_num = err_rnd_num
        self.fn_rnd_num_before = fn_rnd_num_before
        self.fn_rnd_num_after = fn_rnd_num_after
        self.fn_rnd_num_list_before = fn_rnd_num_list_before
        self.fn_rnd_num_list_after = fn_rnd_num_list_after

        self.cache_capture_seed = {}
        self.cache_reproduce = {}
        self.cache_capture = {}

        os.system(f'mkdir -p "{load_cache_dir}"')
        self.load_cache_dir = load_cache_dir

        os.system(f'mkdir -p "{save_cache_dir}"')
        self.save_cache_dir = save_cache_dir

        self.load_cache()

    def moment_short_desc_keys(self):
        return []

    def pair_matching_keys(self):
        return []

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

        return ''

    def refine_json_pred(self, prd_moment, o_json_pred):
        return o_json_pred
        
    def create_moment(self, rnd_num_list, time_no = 1, data = {}):
        return RandifeMoment(self, rnd_num_list, time_no, data)

    def create_moment_list(self, data_df, mpi = 0):
        return RandifeMomentList(self, data_df, mpi)

    def create_moment_pair(self, moment_1, moment_2, data = {}):
        return RandifeMomentPair(self, moment_1, moment_2, data)

    def create_moment_pair_list(self, data_df = None):
        return RandifeMomentPairList(self, data_df)

    def create_simulator(self):
        return RandifeRandomSimulator(self)
        
    def save_cache(self):
        cdir = self.save_cache_dir

        fn = f'{cdir}/rl_cache_capture_seed.pkl'
        with open(fn, 'wb') as f:
            pickle.dump(self.cache_capture_seed, f)

        fn = f'{cdir}/rl_cache_reproduce.pkl'
        with open(fn, 'wb') as f:
            pickle.dump(self.cache_reproduce, f)

        fn = f'{cdir}/rl_cache_capture.pkl'
        with open(fn, 'wb') as f:
            pickle.dump(self.cache_capture, f)

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

    def get_fn_rnd_num_list_before(self):
        return self.fn_rnd_num_list_before

    def get_fn_rnd_num_list_after(self):
        return self.fn_rnd_num_list_after
        
    def get_fn_rnd_num_before(self):
        return self.fn_rnd_num_before

    def get_fn_rnd_num_after(self):
        return self.fn_rnd_num_after

    def get_err_rnd_num(self):
        return self.err_rnd_num

    def err_rnd_num_list(self):
        return [self.err_rnd_num for mix in range(self.moment_size)]
        
    def get_def_rnd_min(self):
        return self.def_rnd_min

    def get_def_rnd_max(self):
        return self.def_rnd_max
        
    def size(self):
        return self.moment_size

    def min(self, mix = 0):
        if mix < 0 or mix >= self.moment_size:
            return self.def_rnd_min
        return self.rnd_min_list[mix]

    def max(self, mix = 0):
        if mix < 0 or mix >= self.moment_size:
            return self.def_rnd_max
        return self.rnd_max_list[mix]

    def set_seed(self, sim_seed):
        random.seed(sim_seed)
        
    def rnd_num(self, mix = 0):
        vmin = self.min(mix)
        vmax = self.max(mix)
        return random.randint(vmin, vmax)

    def rnd_num_list(self):
        num_list = []
        if self.fn_rnd_num_list_before is not None:
            self.fn_rnd_num_list_before(self)
        for ni in range(self.moment_size):
            if self.fn_rnd_num_before is not None:
                self.fn_rnd_num_before(self, ni)
            n = self.rnd_num(ni)
            if self.fn_rnd_num_after is not None:
                self.fn_rnd_num_after(self, ni)
            num_list.append(n)
        if self.fn_rnd_num_list_after is not None:
            self.fn_rnd_num_list_after(self, num_list)
        return num_list

    def has_err_rnd_num(self, rnd_num_list):
        for ni in range(len(rnd_num_list)):
            if rnd_num_list[ni] == self.err_rnd_num:
                return True
        return False

    def reproduce(self, sim_seed, sim_cnt = 1):
        err_rnd_num_list = [self.err_rnd_num for ni in range(self.moment_size)]
        if sim_cnt <= 0:
            return err_rnd_num_list

        key = f'{sim_seed}_{sim_cnt}'
        if key in self.cache_reproduce:
            return self.cache_reproduce[key]
            
        random.seed(sim_seed)
        rnd_num_list = err_rnd_num_list
        for ci in range(sim_cnt):
            rnd_num_list = self.rnd_num_list()

        self.cache_reproduce[key] = rnd_num_list
        
        return rnd_num_list

    def match(self, win_rnd_num_list, prd_rnd_num_list, match_kind = 'match_all'):
        if len(win_rnd_num_list) != self.moment_size or len(prd_rnd_num_list) != self.moment_size:
            return False
        if match_kind == 'match_all':
            for mix in range(self.moment_size):
                if win_rnd_num_list[mix] != prd_rnd_num_list[mix]:
                    return False
            return True
        else:
            return False

    def capture_seed(self, sim_cnt, rnd_num_list):
        lx_rnd_num = [str(x) for x in rnd_num_list]
        key = f'{sim_cnt}_' + '_'.join(lx_rnd_num)
        if key in self.cache_capture_seed:
            return self.cache_capture_seed[key]
            
        sim_seed = 0
        prd_rnd_num_list = self.reproduce(sim_seed, sim_cnt)
            
        while not self.match(rnd_num_list, prd_rnd_num_list, 'match_all'):
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

        random.seed(sim_seed)
        
        sim_cnt = 0
        p = self.rnd_num_list()
        sim_cnt += 1

        while not self.match(win_rnd_num_list, p, 'match_all'):
            p = self.rnd_num_list()
            sim_cnt += 1

        pn = self.reproduce(sim_seed, 1)
        pw = self.reproduce(sim_seed, sim_cnt)
        
        if self.match(pw, win_rnd_num_list, 'match_all') and self.match(pn, prd_rnd_num_list, 'match_all'):
            self.cache_capture[key] = [sim_seed, sim_cnt]
            return sim_seed, sim_cnt
        else:
            self.cache_capture[key] = [-1, -1]
            return -1, -1

    def rnd_num_list_to_str(self, rnd_num_list):
        lx_num_list = [str(x) for x in rnd_num_list]
        return ', '.join(lx_num_list)

    def rnd_num_list_from_str(self, s_num_list):
        lx_num_list = s_num_list.split(', ')
        return [int(x) for x in lx_num_list]

    def rnd_num_list_to_desc(self, rnd_num_list):
        return self.rnd_num_list_to_str(rnd_num_list)

    def rnd_num_list_from_desc(self, s_num_list):
        return self.rnd_num_list_from_str(s_num_list)

class RandifeMoment:
    def __init__(self, rnd_format, rnd_num_list, time_no = 1, data = {}):
        self.rnd_format = rnd_format
        self.time_no = time_no
        self.rnd_num_list = rnd_num_list
        self.win_rnd_num_list = self.rnd_format.err_rnd_num_list()
        self.sim_seed = -1
        self.sim_cnt = -1
        self.data = data

    def capture(self):
        if not self.rnd_format.has_err_rnd_num(self.win_rnd_num_list):
            self.sim_seed, self.sim_cnt = self.rnd_format.capture(self.win_rnd_num_list, self.rnd_num_list)
        else:
            self.sim_seed = self.rnd_format.capture_seed(1, self.rnd_num_list)

    def put_data(self, ndata):
        for key in ndata.keys():
            self.data[key] = ndata[key]
            
    def get_data(self):
        return self.data

    def set_data(self, data):
        self.data = data
        
    def set_data(self, data):
        self.data = data
        
    def get_sim_seed(self):
        return self.sim_seed

    def set_sim_seed(self, sim_seed):
        self.sim_seed = sim_seed

    def get_sim_cnt(self):
        return self.sim_cnt

    def set_sim_cnt(self, sim_cnt):
        self.sim_cnt = sim_cnt
        
    def get_time_no(self):
        return self.time_no

    def set_time_no(self, time_no):
        self.time_no = time_no

    def get_rnd_num_list(self):
        return self.rnd_num_list

    def set_rnd_num_list(self, rnd_num_list):
        self.rnd_num_list

    def get_win_rnd_num_list(self):
        return self.win_rnd_num_list

    def set_win_rnd_num_list(self, win_rnd_num_list):
        self.win_rnd_num_list = win_rnd_num_list

    def to_df(self, mpi = 0):
        if mpi <= 0:
            rw = {'time_no': self.time_no}
            for key in self.data.keys():
                rw[f'{key}'] = self.data[key]
            for mix in range(self.rnd_format.size()):
                no = mix + 1
                rw[f'w_{no}'] = self.win_rnd_num_list[mix]
            for mix in range(self.rnd_format.size()):
                no = mix + 1
                rw[f'n_{no}'] = self.rnd_num_list[mix]
            rw[f'sim_seed'] = self.sim_seed
            rw[f'sim_cnt'] = self.sim_cnt
            return pd.DataFrame([rw])
        else:
            rw = {f'time_no_{mpi}': self.time_no}
            for key in self.data.keys():
                rw[f'{key}_{mpi}'] = self.data[key]
            for mix in range(self.rnd_format.size()):
                no = mix + 1
                rw[f'w_{mpi}_{no}'] = self.win_rnd_num_list[mix]
            for mix in range(self.rnd_format.size()):
                no = mix + 1
                rw[f'n_{mpi}_{no}'] = self.rnd_num_list[mix]
            rw[f'sim_seed_{mpi}'] = self.sim_seed
            rw[f'sim_cnt_{mpi}'] = self.sim_cnt
            return pd.DataFrame([rw])

    def from_df(self, data_df, mpi = 0, rwi = 0):
        if mpi <= 0:
            df = data_df[data_df['time_no'] == self.time_no]
            if len(df) != 1:
                return False
            for key in self.data.keys():
                self.data[key] = df[f'{key}'].iloc[0]
            for mix in range(self.rnd_format.size()):
                no = mix + 1
                self.win_rnd_num_list[mix] = df[f'w_{no}'].iloc[0]
            for mix in range(self.rnd_format.size()):
                no = mix + 1
                self.rnd_num_list[mix] = df[f'n_{no}'].iloc[0]
            self.sim_seed = df[f'sim_seed'].iloc[0]
            self.sim_cnt = df[f'sim_cnt'].iloc[0]
        else:
            df = data_df[data_df[f'time_no_{mpi}'] == self.time_no]
            if len(df) < 1:
                return False
            if rwi < 0 or rwi >= len(df):
                return False
            for key in self.data.keys():
                 self.data[key] = df[f'{key}_{mpi}'].iloc[rwi]
            for mix in range(self.rnd_format.size()):
                no = mix + 1
                self.win_rnd_num_list[mix] = df[f'w_{mpi}_{no}'].iloc[rwi]
            for mix in range(self.rnd_format.size()):
                no = mix + 1
                self.rnd_num_list[mix] = df[f'n_{mpi}_{no}'].iloc[rwi]
            self.sim_seed = df[f'sim_seed_{mpi}'].iloc[rwi]
            self.sim_cnt = df[f'sim_cnt_{mpi}'].iloc[rwi]

        return True

class RandifeMomentList:
    def __init__(self, rnd_format, data_df, mpi = 0):
        self.rnd_format = rnd_format
        self.data_df = None
        self.mpi = mpi
        self.moment_list = []
        self.from_df(data_df)

    def put_data_at(self, mli, ndata):
        self.moment_list[mli].put_data(ndata)
        
    def capture_at(self, mli):
        self.moment_list[mli].capture()
        
    def size(self):
        return len(self.moment_list)

    def at(self, rwi = 0):
        if rwi < 0 or rwi >= self.size():
            return None
        return self.moment_list[rwi]
        
    def time_no_at(self, rwi):
        if rwi < 0 or rwi >= len(self.data_df):
            return None
        if self.mpi <= 0:
            self.data_df[f'time_no'].iloc[rwi]
        else:
            mpi = self.mpi
            self.data_df[f'time_no_{mpi}'].iloc[rwi]

    def moment_by(self, time_no, rwi = 0):
        mpi = self.mpi
        if mpi <= 0:
            df = self.data_df[self.data_df['time_no'] == time_no]
            if len(df) != 1:
                return None
            moment = self.rnd_format.create_moment(self.rnd_format.err_rnd_num_list(), time_no)
            rs = moment.from_df(df, mpi)
            if not rs:
                return None
            return moment
        else:
            df = self.data_df[self.data_df[f'time_no_{mpi}'] == time_no]
            if len(df) == 0:
                return None
            if rwi < 0 or rwi >= len(df):
                return None
            moment = self.rnd_format.create_moment(self.rnd_format.err_rnd_num_list(), time_no)
            rs = moment.from_df(df, mpi, rwi)
            if not rs:
                return None
            return moment
            
    def moment_at(self, rwi, rwi2 = 0):
        if rwi < 0 or rwi >= len(self.data_df):
            return None
        time_no = self.time_no_at(rwi)
        return self.moment_by(time_no, rwi2)

    def add(self, moment):
        if moment is None:
            return
        df = moment.to_df(self.mpi)
        self.moment_list.append(moment)
        if self.data_df is None:
            self.data_df = df
        else:
            self.data_df = pd.concat([self.data_df, df])

    def sort(self, ascending = True, mpi = 0):
        ddf = self.to_df(mpi)
        self.from_df(ddf, ascending)
        
    def from_df(self, ddf, ascending = True):
        if ddf is None:
            return
        self.data_df = None
        self.moment_list = []
        mpi = self.mpi
        if mpi <= 0:
            if ascending:
                ddf = ddf.sort_values(by=[f'time_no'], ascending=[True])
            else:
                ddf = ddf.sort_values(by=[f'time_no'], ascending=[False])                
            l_time_no = list(ddf[f'time_no'].values)
        else:
            if ascending:
                ddf = ddf.sort_values(by=[f'time_no_{mpi}'], ascending=[True])
            else:
                ddf = ddf.sort_values(by=[f'time_no_{mpi}'], ascending=[False])                
            l_time_no = list(ddf[f'time_no_{mpi}'].values)
        l_time_no = [int(x) for x in l_time_no]
        for time_no in l_time_no:
            moment = self.rnd_format.create_moment(self.rnd_format.err_rnd_num_list(), time_no)
            rs = moment.from_df(ddf, mpi)
            if rs:
                self.add(moment)

    def get_df(self):
        return data_df
        
    def to_df(self, mpi = 0):
        adf = None
        sz = self.size()
        for rwi in range(sz):
            moment = self.at(rwi)
            df = moment.to_df(mpi)
            if adf is None:
                adf = df
            else:
                adf = pd.concat([adf, df])
        return adf
        
class RandifeMomentPair:
    def __init__(self, rnd_format, moment_1, moment_2, data = {}):
        self.rnd_format = rnd_format
        self.moment_1 = moment_1
        self.moment_2 = moment_2
        self.data = data
        self.prd_rnd_num_list = self.rnd_format.err_rnd_num_list()

    def put_data(self, ndata):
        for key in ndata.keys():
            self.data[key] = ndata[key]
            
    def get_data(self):
        return self.data

    def get_moment_1(self):
        return self.moment_1

    def get_moment_2(self):
        return self.moment_2

    def set_moment_1(self, moment_1):
        self.moment_1 = moment_1

    def set_moment_2(self, moment_2):
        self.moment_2 = moment_2

    def to_df(self):
        df = self.moment_1.to_df(1)
        df2 = self.moment_2.to_df(2)
        for c in df2.columns:
            df[c] = df2[c]
        for mix in range(self.rnd_format.size()):
            no = mix + 1
            df[f'p_{no}'] = self.prd_rnd_num_list[mix]
        for key in self.data.keys():
            df[key] = self.data[key]
        return df

    def from_df(self, data_df):
        self.moment_1.from_df(data_df, 1)
        self.moment_2.from_df(data_df, 2)
        for mix in range(self.rnd_format.size()):
            no = mix + 1
            self.prd_rnd_num_list[mix] = data_df[f'p_{no}'].iloc[0]
        for key in self.data.keys():
            self.data[key] = data_df[key].iloc[0]

    def simulate(self):
        sim_seed = self.moment_1.get_sim_seed()
        sim_cnt = self.moment_2.get_sim_cnt()
        self.prd_rnd_num_list = self.rnd_format.reproduce(sim_seed, sim_cnt)

        if not self.rnd_format.has_err_rnd_num(self.moment_1.get_win_rnd_num_list()):
            for match_kind in self.rnd_format.pair_matching_keys():
                v = 0
                if self.rnd_format.match(self.moment_1.get_win_rnd_num_list(), self.prd_rnd_num_list, match_kind):
                    v = 1
                self.data[f'{match_kind}'] = v
                
class RandifeMomentPairList:
    def __init__(self, rnd_format, data_df):
        self.rnd_format = rnd_format
        self.data_df = None
        self.pair_list = []
        self.from_df(data_df)

    def put_data_at(self, pli, ndata):
        self.pair_list[pli].put_data(ndata)
        
    def simulate_at(self, pli):
        self.pair_list[pli].simulate()
        
    def size(self):
        return len(self.pair_list)

    def at(self, rwi = 0):
        if rwi < 0 or rwi >= self.size():
            return None
        return self.pair_list[rwi]
        
    def time_no_at(self, rwi):
        if rwi < 0 or rwi >= len(self.data_df):
            return None
        return self.data_df['time_no_1'].iloc[rwi], self.data_df['time_no_2'].iloc[rwi]

    def pair_by(self, time_no_1, time_no_2):
        df = self.data_df[(self.data_df['time_no_1'] == time_no_1)&(self.data_df['time_no_2'] == time_no_2)]
        if len(df) != 1:
            return None
            
        moment_1 = self.rnd_format.create_moment(self.rnd_format.err_rnd_num_list(), time_no_1)
        rs = moment_1.from_df(df, 1)
        if not rs:
            return None

        moment_2 = self.rnd_format.create_moment(self.rnd_format.err_rnd_num_list(), time_no_2)
        rs = moment_2.from_df(df, 2)
        if not rs:
            return None

        pair = self.rnd_format.create_moment_pair(moment_1, moment_2)
        return pair

    def pair_at(self, rwi):
        if rwi < 0 or rwi >= len(self.data_df):
            return None
        time_no_1, time_no_2 = self.time_no_at(rwi)
        return self.pair_by(time_no_1, time_no_2)

    def add(self, pair):
        if pair is None:
            return
        df = pair.to_df()
        self.pair_list.append(pair)
        if self.data_df is None:
            self.data_df = df
        else:
            self.data_df = pd.concat([self.data_df, df])

    def sort(self, ascending_1 = True, ascending_2 = True):
        ddf = self.to_df()
        self.from_df(ddf, ascending_1, ascending_2)
        
    def from_df(self, ddf, ascending_1 = True, ascending_2 = True):
        if ddf is None:
            return

        self.data_df = None
        self.pair_list = []
        
        if ascending_1 is not None and ascending_2 is not None:
            ddf = ddf.sort_values(by=[f'time_no_1', f'time_no_2'], ascending=[ascending_1, ascending_2])
            
        l_time_no_1 = list(ddf[f'time_no_1'].values)
        l_time_no_2 = list(ddf[f'time_no_2'].values)

        for ri in range(len(l_time_no_1)):
            time_no_1 = l_time_no_1[ri]
            time_no_2 = l_time_no_2[ri]

            df = ddf[(ddf['time_no_1'] == time_no_1)&(ddf['time_no_2'] == time_no_2)]
            
            moment_1 = self.rnd_format.create_moment(self.rnd_format.err_rnd_num_list(), time_no_1)
            rs = moment_1.from_df(df, 1)
            if not rs:
                continue
    
            moment_2 = self.rnd_format.create_moment(self.rnd_format.err_rnd_num_list(), time_no_2)
            rs = moment_2.from_df(df, 2)
            if not rs:
                continue
    
            pair = self.rnd_format.create_moment_pair(moment_1, moment_2)
            pair.from_df(df)

            self.add(pair)

    def get_df(self):
        return self.data_df
        
    def to_df(self):
        adf = None
        sz = self.size()
        for rwi in range(sz):
            pair = self.at(rwi)
            df = pair.to_df()
            if adf is None:
                adf = df
            else:
                adf = pd.concat([adf, df])
        return adf

class RandifeRandomSimulator:
    def __init__(self, rnd_format):
        self.rnd_format = rnd_format

    def simulate(self, data_df, prd_time_no, prc_time_cnt, prc_runtime, tck_cnt, has_step_log = False, cache_only = False):
        text = self.rnd_format.heading('simulate', 'method_start')
        print(text)

        prd_moment = self.rnd_format.create_moment(self.rnd_format.err_rnd_num_list(), prd_time_no)
        rs = prd_moment.from_df(data_df, 0)
        if not rs:
            msg = f'Predicting moment [{prd_time_no}] is not found!'
            print(f'== [Error] ==> {msg}')
            try:
                self.rnd_format.save_cache()
            except Exception as e:
                msg = str(e)
                print(f'== [Error] ==> {msg}')
            
            return None, None, None, None, None

        start_time = time.time()
        
        text = self.rnd_format.heading('simulate', 'parameters_start')
        print(text)

        print(f'PRD_TIME_NO: {prd_time_no}')
        print(f'----------')
        prd_moment_data = prd_moment.get_data()
        for key in self.rnd_format.moment_short_desc_keys():
            val = prd_moment_data[key]
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

        prd_moment.capture()
        
        prc_df = data_df[data_df['time_no'] < prd_time_no]
        prc_df = prc_df.sort_values(by=['time_no'], ascending=[False])
        prc_cnt = prc_time_cnt * 2
        if len(prc_df) > prc_cnt:
            prc_df = prc_df[:prc_cnt]
        prc_moment_list = self.rnd_format.create_moment_list(prc_df, 0)

        sz = prc_moment_list.size()
        dbcnt = int(round(sz / 100.0))
        if dbcnt < 1:
            dbcnt = 1
        for li in range(prc_moment_list.size()):
            if time.time() - start_time > prc_runtime:
                break
                
            prc_moment_list.capture_at(li)

            pix = li + 1
            if pix % dbcnt == 0:
                if has_step_log:
                    print(f'== [S1] ==> {pix} / {sz}')

        prc_moment_list.sort(False)
        prc_df = prc_moment_list.to_df(0)
        prc_moment_list.sort(True)

        prc_pair_list = self.rnd_format.create_moment_pair_list()

        sz = prc_time_cnt * prc_time_cnt
        dbcnt = int(round(sz / 100.0))
        if dbcnt < 1:
            dbcnt = 1
        pix = 0
        for pia in range(prc_moment_list.size()):
            if time.time() - start_time > prc_runtime:
                break

            if pia <= prc_time_cnt:
                continue
                
            moment_1 = prc_moment_list.at(pia)

            if not cache_only:
                cnt_dict = {}
                for match_kind in self.rnd_format.pair_matching_keys():
                    cnt_dict[f'{match_kind}_cnt'] = 0
                time_no_dict = {}
            
            for pib in range(prc_moment_list.size()):
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
                    if has_step_log:
                        print(f'== [S2] ==> {pix} / {sz}')
                    
                moment_2 = prc_moment_list.at(pib)

                pair = self.rnd_format.create_moment_pair(moment_1, moment_2)
                prc_pair_list.add(pair)
                prc_pair_list.simulate_at(prc_pair_list.size() - 1)

                if not cache_only:
                    time_key = str(moment_1.get_time_no()) + '_' + str(moment_2.get_time_no())
                    time_no_dict[time_key] = prc_pair_list.size() - 1
                    
                    pair = prc_pair_list.at(prc_pair_list.size() - 1)
    
                    time_key = str(moment_1.get_time_no() - 1) + '_' + str(moment_2.get_time_no())
                    odata = None
                    if key in time_no_dict:
                        pi = time_no_dict[key]
                        odata = prc_pair_list.at(pi).get_data()
                        
                    pdata = pair.get_data()
                    cnt_dict_2 = {}
                    for match_kind in self.rnd_format.pair_matching_keys():
                        cnt_dict[f'{match_kind}_cnt'] = int(cnt_dict[f'{match_kind}_cnt']) + int(pdata[f'{match_kind}'])
                    for match_kind in self.rnd_format.pair_matching_keys():
                        ov = 0
                        if odata is not None:
                            ov = int(odata[f'{match_kind}_cnt'])
                        cnt_dict_2[f'{match_kind}_cnt'] = int(cnt_dict[f'{match_kind}_cnt']) + ov
                    prc_pair_list.put_data_at(prc_pair_list.size() - 1, cnt_dict_2)

        prc_pair_list.sort(False, False)
        
        zdf = prc_pair_list.to_df()
        pdf = zdf[zdf['time_no_1'] == prd_moment.get_time_no() - 1]

        lx_pred = []
        lx_sim_cnt = []
        x_sim_seed = prd_moment.get_sim_seed()
        ma_rsi = - 1
        for ri in range(len(pdf)):
            x_sim_cnt = pdf['sim_cnt_2'].iloc[ri]
            x_p = self.rnd_format.reproduce(x_sim_seed, x_sim_cnt)
            xs_p = self.rnd_format.rnd_num_list_to_str(x_p)
            if not cache_only:
                lx_sim_cnt.append(str(x_sim_cnt))
                lx_pred.append(xs_p)
                if self.rnd_format.match(prd_moment.get_win_rnd_num_list(), x_p, 'match_all'):
                    ma_rsi = ri
            if tck_cnt > 0:
                if ri >= tck_cnt:
                    break

        if cache_only:
            try:
                self.rnd_format.save_cache()
            except Exception as e:
                msg = str(e)
                print(f'== [Error] ==> {msg}')
            
            return None, None, None, None, None
        
        xs_pred = '; '.join(lx_pred)
        xs_sim_cnt = '; '.join(lx_sim_cnt)

        x_w = self.rnd_format.rnd_num_list_to_str(prd_moment.get_win_rnd_num_list())
        x_n = self.rnd_format.rnd_num_list_to_str(prd_moment.get_rnd_num_list())
        json_pred = {'time_no': int(prd_moment.get_time_no()), 'w': x_w, 'n': x_n, 'sim_seed': int(x_sim_seed), 'prc_time_cnt': int(prc_time_cnt), 'tck_cnt': int(tck_cnt), 'sim_cnt': xs_sim_cnt, 'pred': xs_pred, 'ma_rsi': int(ma_rsi)}
        n_json_pred = self.rnd_format.refine_json_pred(prd_moment, json_pred)
        
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

        try:
            self.rnd_format.save_cache()
        except Exception as e:
            msg = str(e)
            print(f'== [Error] ==> {msg}')
        
        return prc_df, json_pred, n_json_pred, zdf, pdf
        
