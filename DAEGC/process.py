import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler,normalize
from typing import Tuple
import torch
import copy

duration_list = [7,15,30,90,180,365,365*3,365*5]
duration_name_list = ['7d','15d','30d','90d','180d','1y','3y','5y']
tname_dir = '../data/tname.xlsx'
personal_info_dir = '../data/personal_info.xlsx'
video_info_dir = '../data/video_info2.csv'



##################################################
def get_tname():
    return pd.read_excel(tname_dir)

# 清洗personal表格
def get_df_person_cl(if_scale=True):
    df_person = pd.read_excel(personal_info_dir,index_col=0)
    df_person_cl = df_person.drop(labels=['uname', 'birthday', 'article_count', 'sign', 'p_name', 'n_name',
                                        'n_condition', 'official_title', 'official_desc', 'attention', 'nid',
                                        'official_veri_type', 'official_veri_desc', 'key_id', 'follower'],axis=1)
    df_person_cl.mid = df_person_cl.mid.astype(int)
    df_person_cl.drop_duplicates(['mid'],inplace=True)
    df_person_cl.reset_index(drop=True)

    # 部分nlevel是na
    df_person_cl.loc[df_person_cl.n_level.isna(), 'n_level'] = 'None'
    # 做onehot处理
    df_person_cl = pd.concat([df_person_cl, pd.get_dummies(df_person_cl['sex'], prefix='sex').iloc[:,1:]], axis=1)
    df_person_cl = pd.concat([df_person_cl, pd.get_dummies(df_person_cl['official_role'], prefix='official_role').iloc[:,1:]], axis=1)
    df_person_cl = pd.concat([df_person_cl, pd.get_dummies(df_person_cl['official_type'], prefix='official_type').iloc[:,1:]], axis=1)
    df_person_cl = pd.concat([df_person_cl, pd.get_dummies(df_person_cl['vip_type'], prefix='vip_type').iloc[:,1:]], axis=1)
    df_person_cl = pd.concat([df_person_cl, pd.get_dummies(df_person_cl['vip_status'], prefix='vip_status').iloc[:,1:]], axis=1)
    df_person_cl = pd.concat([df_person_cl, pd.get_dummies(df_person_cl['n_level'], prefix='n_level').iloc[:,1:]], axis=1)
    # pid只取数量>10的，否则变量数太多
    pidlist = list(df_person_cl.pid.value_counts()[df_person_cl.pid.value_counts()>10].index)
    df_person_cl.loc[~df_person_cl.pid.isin(pidlist),'pid'] = 'others'
    df_person_cl = pd.concat([df_person_cl, pd.get_dummies(df_person_cl['pid'], prefix='pid').iloc[:,1:]], axis=1)
    # 删掉原来的变量
    df_person_cl.drop(labels=['sex','official_role','official_type','vip_type','vip_status','n_level','pid'],inplace=True,axis=1)
    # 对右偏变量取log
    df_person_cl['fans'] = np.log(1+df_person_cl['fans'])
    df_person_cl['friend'] = np.log(1+df_person_cl['friend'])
    df_person_cl['archive_count'] = np.log(1+df_person_cl['archive_count'])
    df_person_cl['like_num'] = np.log(1+df_person_cl['like_num'])
    # 连续变量标准化
    if if_scale:
        scaler = StandardScaler()
        col_to_scale = ['fans','friend','cur_level','archive_count','like_num']
        df_person_cl[col_to_scale] = scaler.fit_transform(df_person_cl[col_to_scale])
    return df_person_cl


##################################################
# 清洗video表格
def get_df_video_cl():
    df_video = pd.read_csv(video_info_dir,index_col=0)
    # 删除没有用的列
    df_video_cl = df_video.drop(labels=['key_id','subtitle','review','is_steins_gate','attribute','Unnamed: 0','is_pay','meta_mid',
                                        'vt','enable_vt','vt_display','playback_position','title','pic', 'copyright','is_live_playback',
                                        'description','author','aid','is_avoided','length','meta_title','is_charging_arc'],axis=1)
    # float to int
    df_video_cl.mid = df_video_cl.mid.astype(int)
    df_video_cl.typeid = df_video_cl.typeid.astype(int)
    # 日期从int变日期
    df_video_cl['created'] = pd.to_datetime(df_video_cl['created'], unit='s')
    df_video_cl['cur_time'] = pd.to_datetime(df_video_cl['cur_time'], unit='s')
    # 重新归纳视频分区
    tname = get_tname()
    tid_dic = {}
    for i in range(tname.shape[0]):
        tid_dic[tname.loc[i,'typeid']] = tname.loc[i,'cat0']
    df_video_cl['cat'] = df_video_cl['typeid'].apply(lambda x: tid_dic.get(x))
    # 计算发布天数
    df_video_cl['datediff'] = (df_video_cl.cur_time - df_video_cl.created).dt.days.astype(int)
    # 标记是否是短视频
    df_video_cl['if_short'] = df_video_cl.length2.apply(lambda x: 1 if x<60 else 0)
    df_video_cl.drop(labels=['cur_time','created','typeid','meta_ep_count'],inplace=True,axis=1)
    return df_video_cl


##################################################
# 特征工程

# comment number
def get_comment_number(df_p,df_v):
    df = pd.DataFrame(df_p['mid'])
    # 每个时间段的comment和
    for i in range(len(duration_list)):
        duration = duration_list[i]
        duration_name = duration_name_list[i]
        df_commenti = pd.DataFrame(df_v.loc[df_v.datediff<=duration,:].groupby(['mid'])['comment'].sum())
        df = pd.merge(df, df_commenti,how='left',left_on='mid',right_on='mid')
        df.loc[df.comment.isna(),'comment'] = 0
        df.rename(columns={'comment':'comment_'+duration_name},inplace=True)
    # 总comment
    df_commenti = pd.DataFrame(df_v.groupby(['mid'])['comment'].sum())
    df = pd.merge(df, df_commenti,how='left',left_on='mid',right_on='mid')
    df.loc[df.comment.isna(),'comment'] = 0
    df.rename(columns={'comment':'comment_ttl'},inplace=True)
    return pd.merge(df_p, df, how='left', left_on='mid', right_on='mid')

# plays
def get_plays(df_p,df_v):
    df = pd.DataFrame(df_p['mid'])
    # 每个时间段的play和
    for i in range(len(duration_list)):
        duration = duration_list[i]
        duration_name = duration_name_list[i]
        df_playi = pd.DataFrame(df_v.loc[df_v.datediff<=duration,:].groupby(['mid'])['play'].sum())
        df = pd.merge(df, df_playi,how='left',left_on='mid',right_on='mid')
        df.loc[df.play.isna(),'play'] = 0
        df.rename(columns={'play':'play_'+duration_name},inplace=True)
    # 总play
    df_playi = pd.DataFrame(df_v.groupby(['mid'])['play'].sum())
    df = pd.merge(df, df_playi,how='left',left_on='mid',right_on='mid')
    df.loc[df.play.isna(),'play'] = 0
    df.rename(columns={'play':'play_ttl'},inplace=True)
    return pd.merge(df_p, df, how='left', left_on='mid', right_on='mid')

# short_num
def get_short_num(df_p,df_v):
    df = pd.DataFrame(df_p['mid'])
    # 每个时间段的short_num和
    for i in range(len(duration_list)):
        duration = duration_list[i]
        duration_name = duration_name_list[i]

        df_short_numi = pd.DataFrame(df_v.loc[(df_v.datediff<=duration)&(df_v.if_short==1),:].groupby(['mid'])['bvid'].count())

        df = pd.merge(df, df_short_numi,how='left',left_on='mid',right_on='mid')
        df.loc[df.bvid.isna(),'bvid'] = 0
        df.rename(columns={'bvid':'short_num_'+duration_name},inplace=True)

    # 总short_num
    df_short_numi = pd.DataFrame(df_v.loc[df_v.if_short==1,:].groupby(['mid'])['bvid'].count())
    
    df = pd.merge(df, df_short_numi,how='left',left_on='mid',right_on='mid')
    df.loc[df.bvid.isna(),'bvid'] = 0
    df.rename(columns={'bvid':'short_num_ttl'},inplace=True)
    return pd.merge(df_p, df, how='left', left_on='mid', right_on='mid')

# long_num
def get_long_num(df_p,df_v):
    df = pd.DataFrame(df_p['mid'])
    # 每个时间段的long_num和
    for i in range(len(duration_list)):
        duration = duration_list[i]
        duration_name = duration_name_list[i]

        df_long_numi = pd.DataFrame(df_v.loc[(df_v.datediff<=duration)&(df_v.if_short==0),:].groupby(['mid'])['bvid'].count())

        df = pd.merge(df, df_long_numi,how='left',left_on='mid',right_on='mid')
        df.loc[df.bvid.isna(),'bvid'] = 0
        df.rename(columns={'bvid':'long_num_'+duration_name},inplace=True)

    # 总long_num
    df_long_numi = pd.DataFrame(df_v.loc[df_v.if_short==0,:].groupby(['mid'])['bvid'].count())
    
    df = pd.merge(df, df_long_numi,how='left',left_on='mid',right_on='mid')
    df.loc[df.bvid.isna(),'bvid'] = 0
    df.rename(columns={'bvid':'long_num_ttl'},inplace=True)
    return pd.merge(df_p, df, how='left', left_on='mid', right_on='mid')

# 视频总长度
def get_length(df_p,df_v):
    df = pd.DataFrame(df_p['mid'])
    # 每个时间段的视频总长度
    for i in range(len(duration_list)):
        duration = duration_list[i]
        duration_name = duration_name_list[i]

        df_length = pd.DataFrame(df_v.loc[(df_v.datediff<=duration),:].groupby(['mid'])['length2'].sum())

        df = pd.merge(df, df_length,how='left',left_on='mid',right_on='mid')
        df.loc[df.length2.isna(),'length2'] = 0
        df.rename(columns={'length2':'length_'+duration_name},inplace=True)

    # 视频总长度
    df_length = pd.DataFrame(df_v.groupby(['mid'])['length2'].sum())
    
    df = pd.merge(df, df_length,how='left',left_on='mid',right_on='mid')
    df.loc[df.length2.isna(),'length2'] = 0
    df.rename(columns={'length2':'length_ttl'},inplace=True)
    return pd.merge(df_p, df, how='left', left_on='mid', right_on='mid')

# video_review
def get_video_review(df_p,df_v):
    df = pd.DataFrame(df_p['mid'])
    # 每个时间段的video_review和
    for i in range(len(duration_list)):
        duration = duration_list[i]
        duration_name = duration_name_list[i]
        df_video_reviewi = pd.DataFrame(df_v.loc[df_v.datediff<=duration,:].groupby(['mid'])['video_review'].sum())
        df = pd.merge(df, df_video_reviewi,how='left',left_on='mid',right_on='mid')
        df.loc[df.video_review.isna(),'video_review'] = 0
        df.rename(columns={'video_review':'video_review_'+duration_name},inplace=True)
    # 总video_review
    df_video_reviewi = pd.DataFrame(df_v.groupby(['mid'])['video_review'].sum())
    df = pd.merge(df, df_video_reviewi,how='left',left_on='mid',right_on='mid')
    df.loc[df.video_review.isna(),'video_review'] = 0
    df.rename(columns={'video_review':'video_review_ttl'},inplace=True)
    return pd.merge(df_p, df, how='left', left_on='mid', right_on='mid')

# 联合投稿数量
def get_union_num(df_p,df_v):
    df = pd.DataFrame(df_p['mid'])
    # 每个时间段的union_num和
    for i in range(len(duration_list)):
        duration = duration_list[i]
        duration_name = duration_name_list[i]

        df_union_numi = pd.DataFrame(df_v.loc[(df_v.datediff<=duration)&(df_v.is_union_video==1),:].groupby(['mid'])['bvid'].count())

        df = pd.merge(df, df_union_numi,how='left',left_on='mid',right_on='mid')
        df.loc[df.bvid.isna(),'bvid'] = 0
        df.rename(columns={'bvid':'union_num_'+duration_name},inplace=True)

    # 总union_num
    df_union_numi = pd.DataFrame(df_v.loc[df_v.is_union_video==1,:].groupby(['mid'])['bvid'].count())
    
    df = pd.merge(df, df_union_numi,how='left',left_on='mid',right_on='mid')
    df.loc[df.bvid.isna(),'bvid'] = 0
    df.rename(columns={'bvid':'union_num_ttl'},inplace=True)
    return pd.merge(df_p, df, how='left', left_on='mid', right_on='mid')

# meta数量
def get_meta_num(df_p,df_v):
    df = pd.DataFrame(df_p['mid'])
    # 每个时间段的meta_num和
    for i in range(len(duration_list)):
        duration = duration_list[i]
        duration_name = duration_name_list[i]

        df_meta_numi = pd.DataFrame(df_v.loc[(df_v.datediff<=duration),['mid','meta_id']].groupby(['mid']).nunique())

        df = pd.merge(df, df_meta_numi,how='left',left_on='mid',right_on='mid')
        df.loc[df.meta_id.isna(),'meta_id'] = 0
        df.rename(columns={'meta_id':'meta_num_'+duration_name},inplace=True)

    # 总meta_num
    df_meta_numi = pd.DataFrame(df_v.loc[:,['mid','meta_id']].groupby(['mid']).nunique())
    
    df = pd.merge(df, df_meta_numi,how='left',left_on='mid',right_on='mid')
    df.loc[df.meta_id.isna(),'meta_id'] = 0
    df.rename(columns={'meta_id':'meta_num_ttl'},inplace=True)
    return pd.merge(df_p, df, how='left', left_on='mid', right_on='mid')

# meta video数量
def get_metavideo_num(df_p,df_v):
    df = pd.DataFrame(df_p['mid'])
    # 每个时间段的metavideo_num和
    for i in range(len(duration_list)):
        duration = duration_list[i]
        duration_name = duration_name_list[i]

        df_metavideo_numi = pd.DataFrame(df_v.loc[(df_v.datediff<=duration)&(~df_v.meta_id.isna()),:].groupby(['mid'])['bvid'].count())

        df = pd.merge(df, df_metavideo_numi,how='left',left_on='mid',right_on='mid')
        df.loc[df.bvid.isna(),'bvid'] = 0
        df.rename(columns={'bvid':'metavideo_num_'+duration_name},inplace=True)

    # 总metavideo_num
    df_metavideo_numi = pd.DataFrame(df_v.loc[(~df_v.meta_id.isna()),:].groupby(['mid'])['bvid'].count())
    
    df = pd.merge(df, df_metavideo_numi,how='left',left_on='mid',right_on='mid')
    df.loc[df.bvid.isna(),'bvid'] = 0
    df.rename(columns={'bvid':'metavideo_num_ttl'},inplace=True)
    return pd.merge(df_p, df, how='left', left_on='mid', right_on='mid')

# 每个分区的视频数量
def get_cat_num(df_p, df_v):
    catlist = get_tname().cat0.unique()
    df = pd.DataFrame(df_p['mid'])
    for cat in tqdm(catlist):
        # 每个时间段的video_num和
        for i in range(len(duration_list)):
            duration = duration_list[i]
            duration_name = duration_name_list[i]

            df_video_numi = pd.DataFrame(df_v.loc[(df_v.datediff<=duration)&(df_v.cat==cat),:].groupby(['mid'])['bvid'].count())

            df = pd.merge(df, df_video_numi,how='left',left_on='mid',right_on='mid')
            df.loc[df.bvid.isna(),'bvid'] = 0
            df.rename(columns={'bvid':'video_num_'+cat+'_'+duration_name},inplace=True)

        # 总video_num
        df_video_numi = pd.DataFrame(df_v.loc[df_v.cat==cat,:].groupby(['mid'])['bvid'].count())
        
        df = pd.merge(df, df_video_numi,how='left',left_on='mid',right_on='mid')
        df.loc[df.bvid.isna(),'bvid'] = 0
        df.rename(columns={'bvid':'video_num_'+cat+'_ttl'},inplace=True)
    return pd.merge(df_p, df, how='left', left_on='mid', right_on='mid')



def get_features(if_scale=True):
    print('loading data...')
    df_person_cl = get_df_person_cl(if_scale)
    df_video_cl = get_df_video_cl()
    
    print('getting features - 1st step...')
    df_ft = get_comment_number(df_person_cl,df_video_cl)
    df_ft = get_plays(df_ft,df_video_cl)
    df_ft = get_short_num(df_ft,df_video_cl)
    df_ft = get_long_num(df_ft,df_video_cl)
    df_ft = get_length(df_ft,df_video_cl)
    df_ft = get_video_review(df_ft,df_video_cl)
    df_ft = get_union_num(df_ft,df_video_cl)
    df_ft = get_meta_num(df_ft,df_video_cl)
    df_ft = get_metavideo_num(df_ft,df_video_cl)
    print('getting features - 2nd step...')
    df_ft = get_cat_num(df_ft, df_video_cl)

    if if_scale:
        print('scaling...')
        for i in range(np.where(df_ft.columns=='comment_7d')[0][0], df_ft.shape[1]):
            df_ft.iloc[:,i] = np.log(1+df_ft.iloc[:,i])
        scaler = StandardScaler()
        df_ft.iloc[:, np.where(df_ft.columns=='comment_7d')[0][0]:df_ft.shape[1]] = scaler.fit_transform(df_ft.iloc[:, np.where(df_ft.columns=='comment_7d')[0][0]:df_ft.shape[1]])
    print('--------DONE--------')

    return df_ft

def get_adj() -> Tuple[np.ndarray,dict]:
    """
    Returns:
    --------
    - Adjacent matrix based on the following relationship
    - Mid dictionary, which contains mid as key and index as value
    """
    print('generating adjacent matrix...')
    df_fl1 = pd.read_csv('../data/following1.csv')
    df_fl2 = pd.read_csv('../data/following2.csv')
    df_ft_mid = get_df_person_cl().reset_index(drop=True).mid
    df_fl = pd.concat([df_fl1[['mid','following_id']],df_fl2[['mid','following_id']]],axis=0).drop_duplicates(subset=['mid'])
    df_fl = df_fl.loc[df_fl.mid.isin(df_ft_mid),:].reset_index(drop=True)
    del(df_fl1,df_fl2)
    mid_list = df_fl.mid.astype(int)
    following_id_list = df_fl.following_id.astype(str)
    # 建立编号字典
    mid_dic = {}
    for i in range(df_fl.shape[0]):
        mid_dic[mid_list[i]] = i
    # 生成邻接矩阵
    adj = np.zeros((df_fl.shape[0],df_fl.shape[0]))
    for i in tqdm(range(df_fl.shape[0])):
        ids = following_id_list[i]
        if ids not in ('forbidden', 'nan'):
            ids = np.array(ids.split(';')).astype(int)
            js = np.vectorize(lambda x: mid_dic.get(x,-1))(ids)
            js = js[js!=-1]
            adj[i,js] = 1
    adj+=np.eye(df_fl.shape[0])

    return(adj, mid_dic, mid_list)


def get_M(adj):
    adj_numpy = adj.cpu().numpy()
    # t_order
    t=2
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    return torch.Tensor(M_numpy)


class myDataSet:
    def __init__(self,if_scale=True) -> None:
        self.adj_label, self.mid_dic, self.mid = get_adj()

        df_feature = get_features(if_scale)
        df_feature = df_feature.loc[df_feature.mid.isin(self.mid),:].reset_index(drop=True) # 删掉Adjacent里没有的id
        df_feature.drop(['mid'],inplace=True,axis=1) # 删掉mid列
        self.x = torch.tensor(df_feature.values, dtype=torch.float)
        self.num_features = df_feature.shape[1]
        self.num_samples = df_feature.shape[0]
        # 对adj归一化并保留最初的adj为adj_label
        self.adj = copy.deepcopy(self.adj_label)
        self.adj = normalize(self.adj, norm="l1")
        # 转换为tensor格式
        self.adj = torch.from_numpy(self.adj).to(dtype=torch.float)
        self.adj_label = torch.from_numpy(self.adj_label).to(dtype=torch.float)
