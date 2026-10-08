# handle_data：无需新增人工或模型标注的精确事件转换

本仓库针对流式视频**事件点定位**（接触、状态转变、实时触发）。上一版把全部 8 个数据集都当作可直接使用，是不符合这个目标的。现改为**每个入选数据集一个独立 Python 模块**，只依赖原有精确标签或同步物理信号。

## 8 个候选中的取舍

| 数据集 | 本轮处理 | 依据 |
| --- | --- | --- |
| Ego4D | **是，核心** | FHO 官方 CONTACT / PNR 关键帧 |
| HD-EPIC | **是，核心** | 官方拾取/放下帧，以及有效的 gaze priming 帧 |
| FEEL | **是，可选** | 同步力信号自动推导接触上升沿；需正确的同步信息、标定阈值或现成接触状态列 |
| MECCANO | 否 | 现有主要是动作区间，不是首次接触关键帧 |
| HoloAssist | 否 | 细粒度行为起止区间不等于物理事件帧 |
| EPIC-KITCHENS-100 | 否 | Action start/end 不能直接变成 first_contact |
| Ego-Exo4D | 否 | Keystep 与 atomic timestamp 不构成精确物理触发时刻 |
| Assembly101 | 否 | 大量 fine-action 区间边界不直接等价于接触帧 |

这里的“免新增标注”是指**不需要在原数据上重新人工逐帧标注、也不调用 VLM/LLM**，并非声称 Ego4D/HD-EPIC 的官方原始标签从未经过人工标注。

## 文件结构

- **ego4d.py**：独立运行；读取官方 FHO hands 的 CONTACT/PNR，可从 fho_main 关联 narration/verb。没有 hands 文件时读取 fho_main 的原始关键帧（如果存在）。
- **hd_epic.py**：独立运行；读取 eye_gaze_priming/priming_info.json 的 object pickup / putdown / gaze priming 帧，自动排除 -1/-2 的无效注视标签。
- **feel.py**：独立运行；读取已同步 force CSV，以现有接触状态列或配置的双阈值滞回规则检测 force_contact_onset。
- **event_common.py**：只负责事件 schema、稳定 ID、JSONL 输出和训练样本模板，**不含任何数据集解析器**。
- **tests/test_selected.py**：独立单元测试。
- **.github/workflows/tests.yml**：自动检查。

原来的 convert_datasets.py（八合一）和对应旧测试已删除。

## 运行方式：不带命令行参数

直接在相应文件顶部 SETTINGS 部分修改数据路径和其他参数，然后运行：

    python ego4d.py

或者：

    python hd_epic.py

或者：

    python feel.py

不要求 argparse，也不用同时下载另外两个数据集。要求 Python 3.10+，基础代码只用标准库。

### Ego4D

在 ego4d.py 配置 ANNOTATION_ROOT、OUTPUT_DIR、VIDEO_ROOT。建议同时具有官方 fho_hands_train.json、fho_hands_val.json 和 fho_main.json。使用 hands 文件时可获得源 split，并尝试按照 video_uid + action_start_frame + action_end_frame 关联 main 里的 narration/verb。未能关联的语义字段留空，绝不猜测。

优先采用来源本身的时间戳；若只有 canonical frame 且有官方名义 FPS，frame/FPS 仅是**名义秒数**，严格的帧级对齐必须验证 PTS。未提供 split 的主文件标为 unspecified，不擅自拆分训练测试。

### HD-EPIC

在 hd_epic.py 配置 ANNOTATION_ROOT、VIDEO_ROOT、FPS_BY_VIDEO / DEFAULT_FPS。官方 priming_info.json 的 start.frame 表示拾取帧，end.frame 表示放下帧；prime_stats.frame_primed 大于等于零才输出 gaze-priming 事件，-1/-2 不会生成正事件。

**拾取不等于 first_contact**；注视预期也不等于拾取。没有经过验证的视频 FPS 时，只输出帧号，不擅自换算秒数（此时也不生成 prospective）。

### FEEL

在 feel.py 配置 DATA_ROOT、OUTPUT_DIR 和同步数据字段名。当前模块针对 force_*aligned*.csv 的 frame_idx、force_row_idx、force 格式；如下载版本不同，需依据真实标注更改对应列名。

可以提供已经存在的接触状态列 LABEL_COLUMN，或者提供传感器标定后的 CONTACT_ON_THRESHOLD、CONTACT_OFF_THRESHOLD。脚本要求先观测稳定非接触，再确认连续接触，才生成 force_contact_onset。模糊读数和缺失信号不会强行当正标签。

**FEEL 的标签是 sensor_derived，不是视觉 first_contact 真值。** 由于 FEEL 同步流原始帧号可能不等于下载的视频切片局部帧号，默认禁用 INCLUDE_PROSPECTIVE；必须核实每段视频的路径、起始偏移、FPS 后再用于流式视频监督。脚本不会伪造这些映射。

## 输出格式

每个模块写各自的 output/ego4d、output/hd_epic、output/feel 目录，内含：

- events.jsonl：包含 source、media、semantics、target_event、evidence、quality、provenance；
- train_instances.jsonl：post_hoc，及在存在时间坐标并被明确启用时的 prospective；
- rejected_samples.jsonl：无法转换的标注；
- stats.json：事件数量、训练样本数量和标签类型。

质量标签：source_exact（官方事件帧）、source_derived_gaze（HD-EPIC 已有的 gaze priming 推导）、sensor_derived（FEEL 物理信号规则）。不同标签类型不能简单合并为 first_contact。

注意：本版不负责下载数据或视频；默认路径要改成本地真实路径，也不会对不存在的视频进行虚假验证。对于正式训练，需要对齐 MP4 PTS 和 session/video clip 的偏移，且不能随机跨同一源视频拆分训练/验证。

## 测试

    python -m compileall -q ego4d.py hd_epic.py feel.py event_common.py tests
    python -m unittest discover -s tests -v

测试通过合成结构样例验证解析、精度标注、门槛/接触滞回与 JSONL 派生；它不等于已经验证完整的官方数据包。CI 自动运行同样检查。

## 官方结构参考

- Ego4D: https://ego4d-data.org/docs/data/annotations-schemas/
- Ego4D FHO: https://ego4d-data.org/docs/benchmarks/hands-and-objects/
- HD-EPIC: https://github.com/hd-epic/hd-epic-annotations
- FEEL: https://www.cs.umd.edu/~edessale/feel
