# handle_data — 流式视频事件标注转换（无需新增人工/模型标注）

这是第一阶段 **annotation-only** 转换器：直接读取公开数据集提供的标注、关键帧或同步传感器数值，统一生成：

- **events.jsonl**：可追溯的事件真值（保留动作区间、源关键帧、时间精度、来源类型）。
- **train_instances.jsonl**：由事件机械派生的后验定位（post_hoc）与有资格的流式触发（prospective）样本。
- **rejected_samples.jsonl**：出错标注的记录；**stats.json**：每个数据集的转换统计。

不会调用 LLM、VLM、视频识别模型，不会补猜“接触帧”，也不会把动作区间起始帧假装成视觉上的首次接触。

## 如何使用

Python ≥ 3.10；默认仅用标准库。若处理 HD-EPIC 的官方 pickle 文件：

    pip install pandas

1. 下载对应数据集的**官方标注**到本地；原始视频可稍后下载。
2. 打开 **convert_datasets.py 顶部 CONFIG**，直接修改：
   - DATA_ROOT：本地数据集根目录
   - DATASET_DIRS：8 个数据集实际存放路径
   - ENABLED：要转换的子集（例如 ["ego4d", "assembly101"]）
   - OUTPUT_ROOT：结果输出目录
   - VIDEO_PATH_TEMPLATES：视频真实路径规则（参见下方）
   - 各数据集的 FPS、传感器阈值等特有参数
3. 在 IDE 运行脚本，或在仓库根目录执行**不带任何参数**的命令：

    python convert_datasets.py

没有输入目录时会明确提示并跳过；这不等于该数据集已成功处理。输入输出均在代码中配置，不使用 argparse。

示例视频路径模板（必须与你下载的文件布局相匹配）：

    VIDEO_PATH_TEMPLATES = {
        "epic_kitchens": "/datadisk/EPIC/videos/{video_id}.MP4",
        "assembly101": "/datadisk/Assembly101/videos/{video_path}",
        "ego4d": "/datadisk/Ego4D/full_scale/{video_id}.mp4",
    }

如果没设置相应的模板，media.video_path 为 null，media.original_video_ref 仍保留原始视频 ID/相对文件名；**不会虚构一个可用视频路径**。同理，脚本不会解码视频或猜测可变帧率（VFR）文件的 PTS。

## 各数据集实际处理内容

| 数据集 | 官方输入文件 / 格式 | 无新增标注的输出 | 注意 |
|---|---|---|---|
| **MECCANO** | RULSTM 5 列、**无表头**的 train/validation/test CSV：video, action, name_action, start, end | 动作区间 | start/end 可是 000012.jpg 形式的抽帧编号；必须确认并设置 MECCANO_ANNOTATION_FPS（代码默认 12，来自常用 RULSTM 抽帧设置，不代表原始视频 FPS）。 |
| **HoloAssist** | 原始 JSON 中的 Fine grained action / Coarse grained action、start/end、attributes | 细粒度动作区间、粗步骤区间及 Verb/Noun/Correctness | 不把 Fine action 的起点解释为首次接触。使用原始标注，不混入处理后模型输出。 |
| **HD-EPIC** | HD_EPIC_Narrations.pkl；eye_gaze_priming/priming_info.json | 动作区间、官方拾取帧 pickup_frame / 放下帧 putdown_frame | pickle 仅在**可信的官方文件**上加载；帧级记录默认不换算秒，除非正确设置 HD_EPIC_INTERACTION_FPS。拾取帧不等同于 first_contact。 |
| **FEEL** | Kitchen/Pxx/force_left_aligned.csv、force_right_aligned.csv，Lab/Pxx/force_*aligned*.csv | 配置传感器阈值后，使用确定性的滞回规则提取 force_contact_onset | 需要**核准实际对齐 FPS 和阈值**，不能使用通用阈值制造“真值”；缺失或损坏的力数据跳过。传感器接触不等于视觉接触。默认不生成此类流式正触发。 |
| **EPIC-KITCHENS-100** | EPIC_100_train.csv、EPIC_100_validation.csv | 动作区间、动词/名词/叙述 | 无真值标注的测试集不会被当作训练标签；不以 start_timestamp 充当接触时刻。 |
| **Ego4D** | 优先 fho_main.json；没有它时回退 fho_hands_*.json | FHO 动作区间、CONTACT → first_contact、PNR → state_change_pnr | 只对官方实际提供的 CONTACT/PNR 生成点事件。用官方 FPS 换算秒只是**名义时间**，评测高精度时间需核对 PTS。 |
| **Ego-Exo4D** | keystep JSON 的 annotations[take_uid].segments | 步骤区间及 scenario/step_name | 原子动作描述的时间锚不是精确关键帧，因此本版只转换 keystep 真值区间。 |
| **Assembly101** | fine-grained-annotations/train/validation/test.csv；coarse-annotations/coarse_labels/*.txt；coarse_splits | 细粒度动作区间、粗步骤区间 | **原视频可为 60fps，但标注帧号按 30fps 抽帧定义**，故起止秒数按 frame/30 计算。不同视角可能来自同一序列，训练/验证不可按单视角随机拆分。 |

这些脚本提供的是**原始标注 → 统一格式的适配器**，没有下载数据集/视频，不保证适配全部社区二次打包格式；有字段不匹配时应对照官方版本处理。

## 标签的精度和训练模式

每条 events 记录中：

    {
      "source": {"dataset": "ego4d", "source_video_id": "..."},
      "semantics": {"coarse_action": "...", "verb": "..."},
      "target_event": {
        "event_type": "first_contact",
        "timestamp_sec": 1.0,
        "frame_idx": 30,
        "interval_start_sec": 0.5,
        "interval_end_sec": 2.5,
        "precision": "frame",
        "time_note": "estimated_from_annotation_fps; verify video PTS"
      },
      "quality": {
        "label_origin": "source_exact",
        "requires_visual_refinement": false
      }
    }

示例中 time_note 明确表示 **1.0 秒是 frame/官方名义 FPS 的估计，不是逐帧校验后的 PTS**。

质量层级：
- source_exact：官方给出关键帧（例 Ego4D CONTACT/PNR，HD-EPIC pickup/putdown）。
- source_interval：官方给出的完整动作/步骤区间，**不能当作精确触发点**。
- sensor_derived：FEEL 力传感器经用户指定阈值自动推得的接触状态上升沿，仍需要独立验证。

post_hoc 可使用明确的点或区间监督。prospective **只从带可用时间坐标、且 label_origin=source_exact 的点事件生成**；可在代码中自行决定是否启用经验证的 sensor_derived 事件（GENERATE_SENSOR_PROSPECTIVE）。训练实例的 response_policy 指明目标发生前保持静默，但**本版不另外生成负样本的视频内容或随机捏造事件阶段**。

## 目录结构

    handle_data/
    ├── convert_datasets.py
    ├── tests/test_converters.py
    ├── .github/workflows/tests.yml
    └── README.md

    output/
    ├── ego4d/
    │   ├── events.jsonl
    │   ├── train_instances.jsonl
    │   ├── rejected_samples.jsonl
    │   └── stats.json
    └── ... 其他数据集同样结构

各数据集**分开存储**：Assembly101 跨视角、EPIC-KITCHENS 与其他衍生数据、同一视频的复用都需要后续以原视频/场景为单位去重、划分，**不能简单拼接所有训练 JSONL 后随机切分**。原始标注源路径、原视频 ID 与 converter 版本保留用于检查和追溯。

## 运行检查

CI 自动执行：

    python -m compileall -q convert_datasets.py tests
    python -m unittest discover -s tests -v

单元测试使用合成的 8 类官方结构近似样例（无需下载大视频），包含区间精度、不虚构 contact、FEEL 滞回、帧率换算与 JSONL 写入。**这不替代实际下载数据后对齐官方版本的端到端测试**。

## 官方结构参考

- MECCANO: https://github.com/fpv-iplab/MECCANO
- HoloAssist: https://holoassist.github.io/data_links/README.html
- HD-EPIC: https://github.com/hd-epic/hd-epic-annotations
- FEEL: https://huggingface.co/datasets/edessa/feel
- EPIC-KITCHENS-100: https://github.com/epic-kitchens/epic-kitchens-100-annotations
- Ego4D: https://ego4d-data.org/docs/data/annotations-schemas/
- Ego-Exo4D: https://docs.ego-exo4d-data.org/annotations/keystep/
- Assembly101: https://github.com/assembly-101/assembly101-annotations
