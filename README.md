# README
这是社交网络挖掘课程项目mxl负责部分

目前主要内容是数据爬虫以及聚类，不定期更新中

## 调用方法
在上传至github时，因为部分文件过大无法上传，所以只上传了personal_info, video_info的sample

特征工程内容请参考文件[process.py](./DAEGC/process.py), 调用方法见文件[test.ipynb](./DAEGC/test.ipynb), 使用时注意将[process.py](./DAEGC/process.py)文件中的对应数据导入路径修改为自己的路径。

本次清洗主要内容为：
* nominal变量进行onehot编码
* pid因为种类过多重新进行了合并，详见SocialNetwork/data/tname.xlsx文件
* 对大部分连续变量取对数，并统一进行标准化化处理
* 引入了rolling 7d, 15d, 30d, 90d, 180d, 1y, 3y, 5y, ttl时间段内的多个特征并归一化处理
最终得到的数据框以mid为主键, 是对up主特征的表示，共9214人，304个特征

## 字段含义

### Personal Information

| key_id             | 内容       |
| ------------------ | ---------- |
| mid                | up主id     |
| uname              | up主名字   |
| sex                | 性别       |
| birthday           | /          |
| fans               | 粉丝数     |
| friend             | 关注数     |
| attention          | 关注数     |
| sign               | 个人签名   |
| cur_level          | 当前等级   |
| pid                | 挂件id     |
| p_name             | 挂件名称   |
| nid                | 勋章id     |
| n_name             | 勋章名称   |
| n_level            | 勋章登记   |
| n_condition        | 勋章条件   |
| official_role      | 认证信息   |
| official_title     | 认证名称   |
| official_desc      | 认证备注   |
| official_type      | 是否认证   |
| official_veri_type | 是否认证   |
| official_veri_desc | 认证信息   |
| vip_type           | 大会员信息 |
| vip_status         | 大会员状态 |
| archive_count      | 视频数量   |
| article_count      | /          |
| follower           | 粉丝数     |
| like_num           | 获赞数     |



### Video Information

| 字段              | 内容               |
| ----------------- | ------------------ |
| key_id            | 主键               |
| mid               | up主编号           |
| comment           | 评论数             |
| typeid            | 分区id             |
| play              | 播放量             |
| pic               | 封面图片           |
| subtitle          | /                  |
| description       | 视频简介           |
| copyright         | 版权类型           |
| title             | 视频标题           |
| review            | /                  |
| author            | 作者名             |
| created           | 发布时间           |
| length            | 视频时长           |
| video_review      | 弹幕数量           |
| aid               | av号               |
| bvid              | bv号               |
| is_pay            | 是否付费视频       |
| is_union_video    | 是否联合投稿       |
| is_steins_gate    | /                  |
| is_live_playback  | 是否直播回放       |
| if_meta           | 是否专题视频       |
| meta_id           | 专题id             |
| meta_title        | 专题标题           |
| meta_ep_count     | 专题视频数量       |
| meta_mid          | 专题作者id         |
| is_avoided        | 是否               |
| attribute         | /                  |
| is_charging_arc   | 是否是充电专属视频 |
| vt                | /                  |
| enable_vt         | /                  |
| vt_display        | /                  |
| playback_position | /                  |
| cur_time          | 当前时间           |
| length2           | 长度分钟           |