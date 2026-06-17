# 会议数据集参考说明

> 本文件仅用于项目内部参考，整理当前用到的会议数据集、主要字段和工程用途，方便后续开发时快速定位输入输出格式。

## 数据集总览

| 数据集 | 链接 | 当前使用 | 原始规模 / 划分 | 关键字段 / 文件 | 工程用途 |
| --- | --- | --- | --- | --- | --- |
| AliMeeting Eval Far | [OpenSLR SLR119](https://www.openslr.org/119/) | 8 场 | 总计 118.75h；Train 212 场 / 104.75h，Eval 8 场 / 4h，Test 20 场 / 10h | `audio_dir/*.wav`，`textgrid_dir/*.TextGrid`；manifest: `dataset`, `split`, `meeting_id`, `audio_uri`, `annotation_uri`, `language` | 中文会议音频端测试：ASR、说话人分离、重叠检测、会议转写 |
| AMI ES2008a | [AMI Download](https://groups.inf.ed.ac.uk/ami/download/) | 1 场 | AMI 总体约 100h；当前只取 `ES2008a.Mix-Headset.wav`，不是官方 unseen eval | `ES2008a.Mix-Headset.wav`；manifest: `dataset`, `split`, `meeting_id`, `audio_uri`, `language` | 英文音频 smoke test，验证 pipeline 可跑英文会议音频 |
| QMSum Test | [QMSum](https://github.com/Yale-LILY/QMSum) | 35 场 | 官方 232 场会议，1,808 个 query-summary pairs；含 train / val / test | `meeting_transcripts`: `speaker`, `content`；`topic_list`；`general_query_list`: `query`, `answer`；`specific_query_list`: `query`, `answer`, `relevant_text_span` | 英文会议问答式摘要评估：长会议文本 → 检索相关片段 → 回答 / 摘要 |
| VCSum Test | [VCSum](https://github.com/hahahawu/VCSum) | 26 场 | 官方 239 场真实中文会议，230h+；含 train / dev / test | `id`, `speaker`, `context`, `summary`, `eos_index`, `discussion`, `agenda`, `highlights` | 中文会议摘要评估：整体摘要、分段摘要、主题边界、高亮信息 |

## 核心字段表

### AliMeeting

| 数据集 | 字段 / 元素 | 类型 | 作用 |
| --- | --- | --- | --- |
| AliMeeting | `audio_dir/*.wav` | 音频文件 | 会议原始音频输入，用于 ASR、说话人分离、重叠检测 |
| AliMeeting | `textgrid_dir/*.TextGrid` | 标注文件 | 时间对齐转写标注，包含说话时间段、文本、说话人信息 |
| AliMeeting | `meeting_id` | 字符串 | 唯一标识一场会议，连接音频和标注 |
| AliMeeting | `audio_uri` / `audio_path` | 路径 | 指向远场 WAV 文件 |
| AliMeeting | `annotation_uri` / `annotation_path` | 路径 | 指向对应 TextGrid 标注 |
| AliMeeting | `language` | 字符串 | 当前为 `zh`，用于选择中文 ASR / 中文后处理 |

### AMI

| 数据集 | 字段 / 元素 | 类型 | 作用 |
| --- | --- | --- | --- |
| AMI | `ES2008a.Mix-Headset.wav` | 音频文件 | 英文会议混合耳麦音频，作为英文音频 pipeline 测试输入 |
| AMI | `meeting_id` | 字符串 | 当前为 `ES2008a`，标识会议样本 |
| AMI | `audio_uri` / `audio_path` | 路径 | 指向 AMI WAV 文件 |
| AMI | `language` | 字符串 | 当前为 `en`，用于选择英文 ASR / 英文后处理 |

### QMSum

| 数据集 | 字段 / 元素 | 类型 | 作用 |
| --- | --- | --- | --- |
| QMSum | `meeting_transcripts` | 列表 | 会议逐轮转写，是问答和摘要的主要输入 |
| QMSum | `speaker` | 字符串 | 当前发言人，用于恢复会议轮次和说话人上下文 |
| QMSum | `content` | 字符串 | 当前轮发言文本，是模型检索和摘要的基础内容 |
| QMSum | `topic_list` | 列表 | 会议主题列表，用于理解会议结构 |
| QMSum | `topic` | 字符串 | 一个会议主题名称 |
| QMSum | `relevant_text_span` | 区间 | 标出某个问题或主题对应的相关发言范围 |
| QMSum | `general_query_list` | 列表 | 通用会议问题，例如“总结这场会议” |
| QMSum | `specific_query_list` | 列表 | 针对某个细节或局部议题的问题 |
| QMSum | `query` | 字符串 | 用户问题 / 评估问题 |
| QMSum | `answer` | 字符串 | 参考答案，也就是模型生成结果的对照标准 |

### VCSum

| 数据集 | 字段 / 元素 | 类型 | 作用 |
| --- | --- | --- | --- |
| VCSum | `id` | 字符串 | 会议样本唯一 ID |
| VCSum | `speaker` | 列表 / 字符串 | 每轮发言对应的说话人 |
| VCSum | `context` | 列表 / 文本 | 中文会议转写内容，是摘要模型的输入 |
| VCSum | `summary` | 字符串 | 整场会议的参考摘要 |
| VCSum | `eos_index` | 列表 | 主题边界位置，用于 topic segmentation |
| VCSum | `discussion` | 列表 | 每个主题段落对应的分段摘要 |
| VCSum | `agenda` | 列表 | 每个主题段落的标题 / headline |
| VCSum | `highlights` | 列表 | 重要句子标记，可用于 salient sentence extraction |

## 使用建议

1. 中文主线优先使用 AliMeeting 和 VCSum。
2. 英文链路先用 AMI 做 smoke test，再切到 QMSum 做摘要和问答评估。
3. 所有真实数据建议放在 `data/` 的 gitignore 路径下，仓库只保留说明文档、脚本和结果摘要。
4. 如果后续增加新数据集，建议继续沿用“总览 + 字段表”的结构，便于统一维护。

