# handle_data：无需新增人工或模型标注的精确事件转换

本仓库针对流式视频**事件点定位**（接触、状态转变、实时触发）。上一版把全部 8 个数据集都当作可直接使用，是不符合这个目标的。现改为**每个入选数据集一个独立 Python 模块**，只依赖原有精确标签或同步物理信号。

## 8 个候选中的取舍

| 数据集 | 本轮处理 | 依据 |
| --- | --- | --- |
| Ego4D | **是，核心** | FHO 官方 CONTACT / PNR 关键帧 |
| HD-EPIC | **是，核心** | 官方拾取/放下帧，以及有效的 gaze priming 帧 |
| FEEL | **仅辅助信号** | 本地 FEEL-Benchmark 的力数据可辅助分析关键帧，当前不生成接触开始/结束真值 |
| MECCANO | 否 | 现有主要是动作区间，不是首次接触关键帧 |
| HoloAssist | 否 | 细粒度行为起止区间不等于物理事件帧 |
| EPIC-KITCHENS-100 | 否 | Action start/end 不能直接变成 first_contact |
| Ego-Exo4D | 否 | Keystep 与 atomic timestamp 不构成精确物理触发时刻 |
| Assembly101 | 否 | 大量 fine-action 区间边界不直接等价于接触帧 |

这里的“免新增标注”是指**不需要在原数据上重新人工逐帧标注、也不调用 VLM/LLM**，并非声称 Ego4D/HD-EPIC 的官方原始标签从未经过人工标注。

## 文件结构

- **ego4d.py**：独立运行；读取官方 FHO hands 的 CONTACT/PNR，可从 fho_main 关联 narration/verb。没有带标签的 hands 文件时流式读取 fho_main 的原始关键帧；`test_unannotated` 不会遮蔽可用标注。
- **hd_epic.py**：独立运行；读取 eye_gaze_priming/priming_info.json 的 object pickup / putdown / gaze priming 帧，自动排除 -1/-2 的无效注视标签。
- **feel.py**：独立运行；本地 LeRobot Parquet 格式只导出辅助力观测。保留旧版已同步 CSV 的显式标签/标定规则接口，与当前 FEEL-Benchmark 输入隔离。
- **event_common.py**：只负责事件 schema、稳定 ID、JSONL 输出和训练样本模板，**不含任何数据集解析器**。
- **tests/**：基本解析测试及数据质量、流式时序、失败输出保护的回归测试。
- **.github/workflows/tests.yml**：自动检查。

原来的 convert_datasets.py（八合一）和对应旧测试已删除。

## 运行方式：不带命令行参数

默认数据路径指向仓库旁的 `datadisk/datasets`，输出写入本仓库的 `output/`，不会修改源数据。Python 3.10+，先安装读取依赖：

    python -m pip install -r requirements.txt

`ijson` 用于流式读取大型 Ego4D JSON，`pyarrow` 用于读取 FEEL Parquet。其他路径或参数可在各文件顶部 SETTINGS 修改，然后运行：

    python ego4d.py

或者：

    python hd_epic.py

或者：

    python feel.py

不要求 argparse，也不用同时下载另外两个数据集。

### Ego4D

在 ego4d.py 配置 ANNOTATION_ROOT、OUTPUT_DIR、VIDEO_ROOT、CLIP_VIDEO_ROOT。优先使用存在的完整视频，否则使用 `v2/clips/<clip_uid>.mp4`。本地只有不带标签的 hands test 文件，因此使用约 2 GB 的 fho_main 流式解析，不将整份 JSON 载入内存。使用带标签的 hands 文件时可获得其 split，并尝试按照 video_uid + action_start_frame + action_end_frame 关联 main 里的 narration/verb。未能关联的语义字段留空，绝不猜测。

优先采用来源本身的时间戳；若只有 canonical frame 且有官方名义 FPS，frame/FPS 仅是**名义秒数**，严格的帧级对齐必须验证 PTS。主文件只有在 clip_uid 与官方 LTA 划分文件精确匹配时才继承该 split，并保留文件路径；这不是 hands benchmark 的划分。未匹配或划分冲突时标为 unspecified，不擅自拆分训练测试。

`fho_main.json` 的 `critical_frames` 支持官方整数帧号形式，hands 文件支持 `frame` 字典。事件真值层保留原视频坐标与片段起点；训练层转换为实际输入 MP4 的局部帧号和秒数，不能把原视频帧号直接作为 clip 答案。例如原帧 18169、片段起始帧 17759，输入片段的答案是第 410 帧；名义时间按 `410 / FPS` 计算，因为来源帧偏移与秒偏移可能相差一帧。原 `clip_critical_frames` 保留为核对证据。一个目录中不能混入多份同名主文件或同一 split 的 hands 文件。关联主文件时若同一动作区间存在多条不同叙述，则不擅自选择其中一条。

### HD-EPIC

在 hd_epic.py 配置 ANNOTATION_ROOT、VIDEO_ROOT、FPS_BY_VIDEO / DEFAULT_FPS。官方 priming_info.json 的 start.frame 表示拾取帧，end.frame 表示放下帧；prime_stats.frame_primed 大于等于零才输出 gaze-priming 事件，-1/-2 不会生成正事件。

视频支持本地 `Videos/<participant>/<video_id>.mp4` 布局。存在 `*_mp4_to_vrs_time_ns.csv` 时按 `mp4_time_ns` 查表换算事件时间；VRS 设备时钟不能作为视频秒数。本地目前有视频及时间映射，但缺少 `priming_info.json`，脚本会明确报缺失标注，不会把映射表变成事件标签。

**拾取不等于 first_contact**；注视预期也不等于拾取。没有经过验证的视频 FPS 时，只输出帧号，不擅自换算秒数（此时也不生成 prospective）。

有效 gaze 帧还必须不晚于对应交互帧，且不早于来源提供的 `prime_window_start`。`priming_info.json` 本身只有物体 ID，没有可直接识别目标物体的自然语言描述；当前这些记录保留在 events.jsonl，暂不派生训练问题。需要可追溯且足以区分目标的物体描述或视觉引用后，才能进入训练实例层，不能把不透明 ID 当作 RGB 视频中的物体名称。

### FEEL

本地目录是 **FeelAuthors/FEEL-Benchmark** 的 LeRobot Parquet 格式，与旧版假设的 aligned CSV 不同。直接运行 `feel.py` 按 `meta/info.json` 和 `meta/episodes.jsonl` 读取声明的 episode，仅写 `output/feel/auxiliary/force_observations.jsonl` 和 `stats.json`，事件数、训练实例数均为零。目录内未被 manifest 声明的额外 MP4 不计入 episode 数。

观测保留 episode 局部帧号、相对时间 `timestamp`、设备时钟 `timestamp_ns`、视频路径及来源 Parquet。来源字段 `action.tactile.left/right.force_total` 实际表示同一只**右手**的两个传感器模块，导出为 `module1_force_raw`、`module2_force_raw`，不能解释成左右手。单位为原始计数，不推断接触状态或跨试次通用阈值；`usable_as_contact_truth` 固定为 false。力信号只作为后续关键帧分析的辅助依据。

旧版 CSV 接口只适用于另行提供、已同步且具有可靠接触状态列或标定阈值的输入，不能用于当前 LeRobot 数据。直接把 LeRobot 数据传给事件转换器会报错。CSV 检测要求先观测稳定非接触，再确认连续接触；缺失同步索引、损坏数值、非连续帧会中断检测，模糊读数不会强行当正标签。`confirmed_at_frame` 保留规则确认时刻，不能声称在线确认延迟为零。

旧版 CSV 接口若被明确配置，其标签仍是 sensor_derived，不是视觉 first_contact 真值，且默认禁用 INCLUDE_PROSPECTIVE；启用需完整视频、同步坐标和 FPS 均已核实。

## 输出格式

事件转换写各自的 output/ego4d、output/hd_epic 目录；旧版 FEEL CSV 接口显式配置后也使用同样格式。本地 FEEL-Benchmark 使用上面的独立辅助输出。事件输出内含：

- events.jsonl：包含 source、media、semantics、target_event、evidence、quality、provenance；
- train_instances.jsonl：有目标描述的 post_hoc，及在存在时间坐标并被明确启用时的 prospective；
- rejected_samples.jsonl：无法转换的标注；
- stats.json：事件数量、训练样本数量和标签类型。

质量标签：source_exact（官方事件帧）、source_derived_gaze（HD-EPIC 已有的 gaze priming 推导）、sensor_derived（FEEL 物理信号规则）。不同标签类型不能简单合并为 first_contact。

schema / converter 版本为 2.2。FPS 换算记录 `fps_source`，训练答案保留时间精度说明。`input.video_id` 标识实际输入文件，`input.source_video_id` 用于同源视频分组；`input.time_basis` 为 video_file。缺失视频的事件仍保留，计入 `events_without_available_media`，不派生训练实例。训练问题使用来源动作描述、名词或明确的左右手信息；缺少目标描述的事件计入 `events_without_grounded_query`，不会虚增训练规模。同一模式、视频输入范围和问题若有不同答案或 split，所有冲突实例都会被排除并计入 `instances_with_ambiguous_query`；相同 split、问题和答案的重复实例只保留一条，并计入 `duplicate_training_instances`。模板仍需下游核对未标注的同类事件是否导致目标不唯一。

prospective 要求查询严格早于事件，且不早于源 context 起点；零时刻事件或没有可用事前范围的事件只保留 post_hoc。流从 `playback_start_sec` 开始，监督契约明确只允许访问当前时刻及之前的帧。`answer` / `response_policy` 是监督数据，不应送入模型输入；这里生成的是任务描述，执行逐帧播放和限制未来帧访问仍由下游训练器负责。

stats 分别统计源事件、训练实例和标签来源，`cross_split_videos` 列出跨 split 的同源视频，保留官方划分而不擅自调整。正式训练或无泄漏评测前必须按原视频分组处理这些重叠；报告并不意味着已消除泄漏。同一 event_id 的标签或 split 冲突会报错。转换完成前使用临时文件，中途解析失败不会覆盖上一次成功的四个输出文件。

本版不负责下载数据或视频。对于正式训练，需要核实 MP4 PTS，且不能随机跨同一源视频拆分训练/验证。本地全量转换和视频核对结果见 [验证记录](validation/README.md)；训练模板数量不等于已经完成视觉语义核验的独立关键帧数量。

## 测试

    python -m compileall -q ego4d.py hd_epic.py feel.py event_common.py tests
    python -m unittest discover -s tests -v

测试通过合成及本地结构回归样例验证解析、片段坐标、精度、接触滞回、辅助力信号隔离与 JSONL 派生。CI 安装读取依赖并运行同样检查。真实数据核对的范围和限制另见验证记录。

## 官方结构参考

- Ego4D: https://ego4d-data.org/docs/data/annotations-schemas/
- Ego4D FHO: https://ego4d-data.org/docs/benchmarks/hands-and-objects/
- HD-EPIC: https://github.com/hd-epic/hd-epic-annotations
- 本地 FEEL-Benchmark：`datadisk/datasets/FEEL/README.md`（来源、传感器模块及时间字段说明）
