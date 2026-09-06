# 参考文献核验台账（第一轮）

日期：2026-09-05。范围：原 `refs.bib` 的 17 条。元数据核验、全文阅读、逐句支持度是不同状态；本文件不等于全量引用通过，也不证明作者本人已阅读全文。尚未核验的出版字段不会进入新稿的最终 bibliography。

以下“原记录”用于留痕，可能包含待修错误；“处理”给出已查证差异。网页存在不等于支持本文所有推论。

## anderson2018vision
- 原题名：Vision-and-Language Navigation: Interpreting Visually-Grounded Navigation Instructions in Real Environments
- 原作者：Anderson, Peter and Wu, Qi and Teney, Damien and Bruce, Jake and Johnson, Mark and S{\"u}nderhauf, Niko and Reid, Ian and Gould, Stephen and van den Hengel, Anton
- 原 venue/year：Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition；2018
- 核验状态：已核验主要元数据
- 标识：arXiv:1711.07280；本稿引用 CVPR 2018 版本
- 来源：[权威/作者来源](https://openaccess.thecvf.com/content_cvpr_2018/html/Anderson_Vision-and-Language_Navigation_Interpreting_CVPR_2018_paper.html)
- 支持范围：支持经典语言导航任务背景，不支持航空复核性能。
- 处理：作者、标题、会议、页码匹配。

## gupta2019xbd
- 原题名：{xBD}: A Dataset for Assessing Building Damage
- 原作者：Gupta, Ritwik and Hosfelt, Richard and Sajeev, Sandra and Patel, Nirav and Goodman, Bryce and Doshi, Jigar and Heim, Eric and Choset, Howie and Gaston, Matthew
- 原 venue/year：Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops；2019
- 核验状态：存在已核验；原条目需修正
- 标识：arXiv:1911.09296
- 来源：[权威/作者来源](https://arxiv.org/abs/1911.09296)
- 支持范围：支持灾前/灾后建筑损伤数据来源。
- 处理：原条目混合了 arXiv 的九位作者与 CVPRW venue，标题也不完整。可选完整 arXiv 版本，或另核 CVPRW 的 Creating xBD 版本（作者及顺序不同）。不要混用。

## bajcsy1988active
- 原题名：Active Perception
- 原作者：Bajcsy, Ruzena
- 原 venue/year：Proceedings of the IEEE；1988
- 核验状态：存在已核验；卷期 DOI 复核待完成
- 标识：待 publisher 复核；暂不补 DOI
- 来源：[权威/作者来源](https://people.eecs.berkeley.edu/~yang/courses/cs294-6/papers/Bajcsy.active%20Perception.pdf)
- 支持范围：支持主动感知经典概念。
- 处理：作者存档论文支持存在；正式 bibliography 前再核 IEEE 元数据。

## guo2017calibration
- 原题名：On Calibration of Modern Neural Networks
- 原作者：Guo, Chuan and Pleiss, Geoff and Sun, Yu and Weinberger, Kilian Q.
- 原 venue/year：Proceedings of the 34th International Conference on Machine Learning；2017
- 核验状态：已核验主要元数据
- 标识：PMLR 70:1321–1330
- 来源：[权威/作者来源](https://proceedings.mlr.press/v70/guo17a.html)
- 支持范围：支持温度缩放/概率校准背景，不保证域外或 VLM 自报置信可靠。
- 处理：补齐 volume=70。

## liu2023aerialvln
- 原题名：{AerialVLN}: Vision-and-Language Navigation for {UAV}s
- 原作者：Liu, Shubo and Zhang, Hongsheng and Qi, Yuankai and Wang, Peng and Zhang, Yanning and Wu, Qi
- 原 venue/year：Proceedings of the IEEE/CVF International Conference on Computer Vision；2023
- 核验状态：已核验主要元数据
- 标识：官方 CVF 记录；DOI 暂不添加
- 来源：[权威/作者来源](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_AerialVLN_Vision-and-Language_Navigation_for_UAVs_ICCV_2023_paper.html)
- 支持范围：支持连续空间航空 VLN 基准。
- 处理：原作者、年份与页码匹配。不能直接移用其成功率与本项目比较。

## lee2025citynav
- 原题名：{CityNav}: A Large-Scale Dataset for Real-World Aerial Navigation
- 原作者：Lee, Jungdae and Miyanishi, Taiki and Kurita, Shuhei and Sakamoto, Koya and Azuma, Daichi and Matsuo, Yutaka and Inoue, Nakamasa
- 原 venue/year：Proceedings of the IEEE/CVF International Conference on Computer Vision；2025
- 核验状态：已核验主要元数据
- 标识：官方 CVF 记录
- 来源：[权威/作者来源](https://openaccess.thecvf.com/content/ICCV2025/html/Lee_CityNav_A_Large-Scale_Dataset_for_Real-World_Aerial_Navigation_ICCV_2025_paper.html)
- 支持范围：支持真实城市来源的航空导航、地理空间任务背景。
- 处理：正式 ICCV 标题与旧 docs 中题名有差异；采用出版社版本。真实城市扫描环境不等于本文二维卫星重渲染。

## bircher2016nbv
- 原题名：Receding Horizon ``Next-Best-View'' Planner for 3D Exploration
- 原作者：Bircher, Andreas and Kamel, Mina and Alexis, Kostas and Oleynikova, Helen and Siegwart, Roland
- 原 venue/year：IEEE International Conference on Robotics and Automation；2016
- 核验状态：存在及主要元数据已核验；页码再核
- 标识：10.1109/ICRA.2016.7487281
- 来源：[权威/作者来源](https://www.research-collection.ethz.ch/entities/publication/ae2aefd8-1e86-4601-bb36-fa175be18481)
- 支持范围：支持滚动时域下一视点探索的先行工作。
- 处理：不能把本文查表熵策略视作已经复现了该三维探索方法。

## chen2022bit
- 原题名：Remote Sensing Image Change Detection with Transformers
- 原作者：Chen, Hao and Qi, Zipeng and Shi, Zhenwei
- 原 venue/year：IEEE Transactions on Geoscience and Remote Sensing；2022
- 核验状态：存在/作者/DOI 已核验；正式卷年待最终核对
- 标识：10.1109/TGRS.2021.3095166
- 来源：[权威/作者来源](https://github.com/justchenhao/BIT_CD)
- 支持范围：支持遥感双时相变化检测架构背景。
- 处理：作者官方仓库给 2021 在线版本；原 bib 为 2022 卷 60。区别在线年与正式卷年，不自行认定一年必错。

## bandara2022changeformer
- 原题名：{A} Transformer-Based Siamese Network for Change Detection
- 原作者：Bandara, Wele Gedara Chaminda and Patel, Vishal M.
- 原 venue/year：IEEE International Geoscience and Remote Sensing Symposium；2022
- 核验状态：存在/作者/会议年已核验；页码 DOI 待 publisher 复核
- 标识：arXiv:2201.01293；IEEE document 9883686
- 来源：[权威/作者来源](https://github.com/wgcban/ChangeFormer)
- 支持范围：支持 Transformer Siamese 变化检测先行工作。
- 处理：作者官方仓库确认 IGARSS 2022；不把二元变化检测与四类损伤分级混为同一任务。

## fang2022snunet
- 原题名：{SNUNet-CD}: A Densely Connected Siamese Network for Change Detection of {VHR} Images
- 原作者：Fang, Sheng and Li, Kaiyu and Shao, Jinyuan and Li, Zhe
- 原 venue/year：IEEE Geoscience and Remote Sensing Letters；2022
- 核验状态：存在/作者已核验；出版细节待复核
- 标识：DOI/arXiv 尚未核实，不补写
- 来源：[权威/作者来源](https://likyoo.github.io/)
- 支持范围：支持密集连接 Siamese 变化检测背景。
- 处理：作者主页确认题名及作者；卷、年、页码需出版社进一步核验。

## ovadia2019can
- 原题名：Can You Trust Your Model's Uncertainty? {E}valuating Predictive Uncertainty Under Dataset Shift
- 原作者：Ovadia, Yaniv and Fertig, Emily and Ren, Jie and Nado, Zachary and Sculley, D. and Nowozin, Sebastian and Dillon, Joshua and Lakshminarayanan, Balaji and Snoek, Jasper
- 原 venue/year：Advances in Neural Information Processing Systems；2019
- 核验状态：存在及作者/会议年已核验
- 标识：NeurIPS 2019 / volume 32
- 来源：[权威/作者来源](https://papers.nips.cc/paper/9547-can-you-trust-your-models-uncertainty-evaluating-predictive-uncertainty-under-dataset-shift.pdf)
- 支持范围：支持数据分布改变下不确定性评估的必要性。
- 处理：不能据它直接证明本模型的跨事件失败原因。

## ren2023knowno
- 原题名：Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners
- 原作者：Ren, Allen Z. and Dixit, Anushri and Bodrova, Alexandra and Singh, Sumeet and Tu, Stephen and Brown, Noah and Xu, Peng and Takayama, Leila and Xia, Fei and Varley, Jack and Xu, Zhenjia and Sadigh, Dorsa and Zeng, Andy and Majumdar, Anirudha
- 原 venue/year：Conference on Robot Learning；2023
- 核验状态：已核验主要元数据
- 标识：PMLR 229:661–682
- 来源：[权威/作者来源](https://proceedings.mlr.press/v229/ren23a.html)
- 支持范围：支持不确定性对齐与请求帮助/选择性行为背景。
- 处理：补卷页；其保证依赖自身方法与假设，不自动转移到本文。

## zhang2025citynavagent
- 原题名：{CityNavAgent}: Aerial Vision-and-Language Navigation with Hierarchical Semantic Planning and Global Memory
- 原作者：Zhang, Weichen and Gao, Chen and Yu, Shiquan and Peng, Ruiying and Zhao, Baining and Zhang, Qian and Cui, Jinqiang and Chen, Xinlei and Li, Yong
- 原 venue/year：Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)；2025
- 核验状态：已核验主要元数据
- 标识：10.18653/v1/2025.acl-long.1511
- 来源：[权威/作者来源](https://aclanthology.org/2025.acl-long.1511/)
- 支持范围：支持分层语义规划与全局记忆的明确来源。
- 处理：当前作者及页码匹配；应说明继承与改动，不能将其模块独立包装为原创。

## gao2024stmr
- 原题名：Exploring Spatial Representation to Enhance {LLM} Reasoning in Aerial Vision-Language Navigation
- 原作者：Gao, Yunpeng and Wang, Zhigang and Han, Pengfei and Jing, Linglin and Wang, Dong and Zhao, Bin
- 原 venue/year：arXiv preprint arXiv:2410.08500；2024
- 核验状态：已核验 arXiv 元数据
- 标识：arXiv:2410.08500，v1 2024-10-11，v3 2025-08-11
- 来源：[权威/作者来源](https://arxiv.org/abs/2410.08500)
- 支持范围：支持语义拓扑度量表示与 LLM 空间推理。
- 处理：2024 是首发年；实际使用版本应固定。不得将 2025 修订全文误称已核验 2024 原版本全部细节。

## shah2023lmnav
- 原题名：{LM-Nav}: Robotic Navigation with Large Pre-Trained Models of Language, Vision, and Action
- 原作者：Shah, Dhruv and Osinski, Blazej and Levine, Sergey
- 原 venue/year：Proceedings of The 6th Conference on Robot Learning；2023
- 核验状态：存在已核验；原作者表错误
- 标识：PMLR 205:492–504
- 来源：[权威/作者来源](https://proceedings.mlr.press/v205/shah23b.html)
- 支持范围：支持预训练模型与图式导航组合的先行工作。
- 处理：正确作者：Dhruv Shah; Błażej Osiński; Brian Ichter; Sergey Levine。原 bib 漏 Brian Ichter，并需恢复姓名重音。PMLR 出版年 2023 与 CoRL 2022 举办年应区分。

## rahnemoonfar2023rescuenet
- 原题名：{RescueNet}: A High Resolution {UAV} Semantic Segmentation Dataset for Natural Disaster Damage Assessment
- 原作者：Rahnemoonfar, Maryam and Chowdhury, Tashnim and Murphy, Robin
- 原 venue/year：Scientific Data；2023
- 核验状态：存在/期刊/文章号已核验；作者表待 publisher 正文复核
- 标识：10.1038/s41597-023-02799-4
- 来源：[权威/作者来源](https://www.nature.com/articles/s41597-023-02799-4)
- 支持范围：支持真实灾后 UAV 语义分割数据来源。
- 处理：出版社检索结果确认文章，正文抓取失败；不能仅用其数据集引用的作者表代替文章作者核验。原 bib 三位作者暂作待核。

## zhang2026esarbench
- 原题名：{ESARBench}: A Benchmark for Agentic {UAV} Embodied Search and Rescue
- 原作者：Zhang, Daoxuan and Chen, Ping and Zhou, Jianyi and Yang, Shuo
- 原 venue/year：arXiv；2026
- 核验状态：已核验 arXiv 元数据；引用语境需改
- 标识：arXiv:2605.01371，2026-05-02 v1
- 来源：[权威/作者来源](https://arxiv.org/abs/2605.01371)
- 支持范围：支持具身 UAV 搜救评测背景。
- 处理：其任务和空间记忆/规划诊断不能独立验证本项目的损伤跨事件原因，更不支持“而非平台或传感”的排除式归因。标注预印本。

## 新检索候选与期刊写作样例

- Jinwei Hu, Yi Dong, Zhengtao Ding, Xiaowei Huang, *Enhancing robustness of LLM-driven multi-agent systems through randomized smoothing*. [CJA 出版社记录](https://www.sciencedirect.com/science/article/pii/S1000936125003851)：39(7), 103779, 2026；DOI 10.1016/j.cja.2025.103779。在线年份 2025 与正式卷年 2026 不同。本地 MinerU 抽取即该文，可参考问题—方法—仿真组织；不是本文单 UAV 复核方法的直接证据，不移植安全证明或多智能体贡献。

- HTNav 的 [CVPR 2026 官方 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Fan_HTNav_A_Hybrid_Navigation_Framework_with_Tiered_Structure_for_Urban_CVPR_2026_paper.pdf) 已发现，待全文与方法可比性审阅；暂不进入 bibliography。

## 下一轮检索

围绕最相关方法补主动感知的任务价值、有限预算观测、航空 VLA/基础模型、灾情智能体与选择性预测。旧 docs 中的 AerialClaw、RescueADI、OpenFly 等只作检索线索，不能直接从笔记复制入参考文献。须记录具体来源、模型/任务/观测条件及与本工作的差异。此为早期检索状态；22 主题的后续定向覆盖见 literature_topic_coverage.md，仍不称穷尽性系统综述。

## 英文稿 bibliography 更新（2026-09-05）

新稿选择 17 条具体支持正文的引用；原 17 条逐条核验记录保留作为历史台账。BibTeX 的 `sourceurl` 保留来源指针，避免长 URL 影响印刷排版。没有从论文数量推出检索完整性。

- `gupta2019xbd`：改用 [arXiv:1911.09296](https://arxiv.org/abs/1911.09296) 的完整题名和九作者；不与 CVPRW 的不同题名/作者版本混用。
- `shah2023lmnav`：按 [PMLR](https://proceedings.mlr.press/v205/shah23b.html) 补 Brian Ichter，卷205、492–504、正式卷年2023。
- `zhang2025citynavagent`：加入 ACL 官方 DOI 10.18653/v1/2025.acl-long.1511。
- `gao2026openfly`：采用 [官方 v7](https://arxiv.org/abs/2502.18041v7)，题名 *A comprehensive platform for aerial vision-language navigation*，2026版，23作者末尾为 Dong Wang / Xuelong Li / Zhigang Wang / Bin Zhao。正文仅用其平台、数据和多高度特点，不照搬性能结论。
- `li2026aerialclaw`：11作者和 DOI 按 [arXiv:2606.12142](https://arxiv.org/abs/2606.12142) 核验。用于限定模块化闭环架构并非本稿原创。
- `liu2024rescueadi`：Zhuoran Liu、Danpei Zhao、Bo Yuan；采用 [2024 arXiv](https://arxiv.org/abs/2410.13384)，没有填入未核实的 TGRS 卷页。用于比较灾情工具智能体任务。
- `curtis2024tampura`：七作者、RSS 2024、DOI 10.15607/RSS.2024.XX.118 来自 [RSS 官方条目](https://www.roboticsproceedings.org/rss20/p118.html)。官方给出的 BibTeX 未给页码，保留空页字段导致的非致命样式提示，不编造范围。
- `geifman2019selectivenet`：按 [PMLR](https://proceedings.mlr.press/v97/geifman19a.html) 记录两作者、卷97、2151–2159，支持风险/覆盖概念。
- `ovadia2019can`：题名、九作者、NeurIPS32、2019 均按 [官方记录](https://papers.nips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html) 核实。页范围 13991–14002 在其他论文参考中出现，但直接官方 BibTeX 下载未解析；本稿暂不填页码，并保留样式提示。

其余保留条目为 Anderson、AerialVLN、CityNav、STMR、Bajcsy、Guo、KnowNo。Bajcsy 未添加尚未直接核对的 DOI。ChangeFormer/BIT/SNUNet/RescueNet/ESARBench/next-best-view 等原条目因不直接支撑当前主文而不强行保留；并非认定这些论文无效。CJA 样例只作组织参考，不强加无关引用。

正文对上述论文的技术比较限于已核实的范围。当前没有正式全文查重报告、PRISMA筛选记录或全领域新颖性证明。

补充 `romano2020adaptive`：按 [NeurIPS 官方记录](https://proceedings.neurips.cc/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html)、[作者仓库](https://github.com/msesia/arc) 与 [arXiv 记录](https://arxiv.org/abs/2006.02544) 核实三作者、题名、NeurIPS33/2020，补充 APS 方法来源。未继承其覆盖保证；页码暂不填，BibTeX 缺页提示总计三条。


## 新增直接近邻（2026-09-06）

`zhao2025cityeqa`：*CityEQA: A Hierarchical LLM Agent on Embodied Question Answering Benchmark in City Space*；Yong Zhao, Kai Xu, Zhengqiu Zhu, Yue Hu, Zhiheng Zheng, Yingfeng Chen, Yatai Ji, Chen Gao, Yong Li, Jincai Huang；EMNLP 2025，12465–12480；DOI 10.18653/v1/2025.emnlp-main.630。来源为 [ACL 官方元数据](https://aclanthology.org/2025.emnlp-main.630/)，方法比较另参作者 arXiv v1。用于 Introduction 与 Related work，承认主动城市问答及视角细化已有直接先例。当前正文共有 **18 条引用**。未声称完整读取正式出版 PDF。
