# 本地真实数据验证记录

日期：2026-10-10。数据根目录：`/home/cc/ccx/datadisk/datasets`，源数据只读。机器可读结果见 [local_data_report.json](local_data_report.json)。

## 修复及结果

| 数据集 | 本地输入与修复 | 实际输出 |
| --- | --- | --- |
| Ego4D | 忽略不带标签的 hands test；流式读取大型 main；支持已下载的 clip；保留原视频真值，训练答案转换为输入片段坐标 | 172141 个源事件，344029 条训练模板，分布于 train / val / unspecified |
| HD-EPIC | 支持参与者子目录视频；使用 MP4 时间映射而非 VRS 设备时钟 | 156 个视频与 156 个映射文件，本地缺少 priming_info.json，未生成事件标签 |
| FEEL-Benchmark | 按 episode manifest 读取 Parquet；区分右手两个传感器模块；保留相对时间和设备时钟 | 52 个试次、3317 个 episode、101219 条辅助力观测；0 个接触真值、0 条训练标签 |

Ego4D 事件包含 59182 个 CONTACT 和 112959 个 PNR，不将它们合并为同一种物理事件。派生模板包含 172063 条 post_hoc 与 171966 条 prospective；344029 是全部划分的模板总量，不能视为 344029 个独立事件或纯训练集大小。

排除了 70 个缺视频事件的训练派生、89 条重复模板和 20 条答案或划分冲突模板。缺视频的事件仍保留在来源层。转换失败时先前输出不会因解析错误被覆盖。

FEEL 目录实际有 3345 个 MP4，只有 manifest 声明的 3317 个 episode 计入规模。原始字段中的 left/right 表示同一右手的模块，不代表左右手；原始力计数不能自动生成接触开始/结束真值。

## 验证范围

- 全量转换 Ego4D 标注；对全部可用事件核对 `canonical_frame - clip_parent_start_frame` 与官方 `clip_critical_frames`，差异为零。
- 用 ffprobe 检查全部 2472 个被事件引用的实际片段：名义 FPS 与实际 FPS 匹配，时间起点为零，目标帧未越界。
- 逐条核对全部 344029 条派生模板：答案帧号/秒数位于实际视频内，prospective 查询严格早于目标；post_hoc 范围允许来源片段边界与实际时长之间至多一帧的舍入差。
- 在零偏移和非零偏移片段中各解码一个视频前缀，核对目标帧 PTS 与局部名义秒数；误差约 0.000000333 秒。具体帧号、路径和范围保存在 JSON 记录中。
- 检查全部 156 个 HD-EPIC 视频头及全部映射行，映射帧数与 MP4 一致，MP4 时间序列正常。
- 全量读取 3317 个 FEEL episode Parquet，检查帧索引连续及行数符合 manifest。101219 条观测的相对时间、设备时钟均非负且在 episode 内递增，相对时间与来源名义 FPS 一致。
- 每个 FEEL 试次抽查一个 manifest 视频，共 52 个：视频帧数与 episode 声明一致，FPS 与来源名义值的差小于 0.001。
- Python 3.13 的 41 项测试全部通过；Python 3.10 通过 40 项，因未安装可选 Parquet 依赖跳过 1 项。编译及 `git diff --check` 通过。

运行入口仍为 `python ego4d.py`、`python feel.py`，事先安装 `requirements.txt`。本次验证将 ijson 3.5.1、pyarrow 26.0.0 安装到 `/tmp/handle-data-validation-deps`，通过 PYTHONPATH 使用，未更改已有 Python 环境。视频头检查使用 `ffprobe -select_streams v:0 -show_entries stream=avg_frame_rate,nb_frames,duration,start_time`；抽样 PTS 使用 `-read_intervals 0%+<秒数> -show_frames -show_entries frame=best_effort_timestamp_time`。

## 尚未满足的训练与评测条件

Ego4D 主文件按精确 clip UID 匹配继承官方 LTA split，并保留划分来源；这不是 hands benchmark 的 train/val 划分。发现 **216 个源视频跨 train/val**，完整名单在 `output/ego4d/stats.json` 的 `cross_split_videos`。当前保留来源划分，正式训练与无泄漏评测必须先按源视频分组处理重叠，不能直接把这份划分声称为无泄漏评测集。

视频头全量检查及两个解码前缀不等于全量视频解码、逐帧 PTS 或视觉语义核验。模板不能证明事前范围内没有其他未标注的同类事件，也不能证明模型已具备在线触发能力；未来帧访问限制仍需由训练器执行。FEEL 的检查确认文件坐标和时钟结构，未重新标定传感器或验证物理接触语义。

本地 HD-EPIC 尚缺官方事件标注；即使补齐 priming_info.json，当前物体 ID 本身也不是 RGB 视频中可识别的目标描述。缺少可追溯描述或视觉引用时仅保留事件来源记录，不虚构训练问题。
