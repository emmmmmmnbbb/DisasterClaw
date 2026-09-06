# 文献研究记录与研究缺口约束

更新：2026-09-06。使用 ARS 的检索、来源核验和论断对齐流程。当前为定向多轮检索；不是 PRISMA 系统综述，不报告虚构筛选数量或全领域穷尽性。

## 已核实的近邻与对主线的影响

**航空语言导航。** AerialVLN、CityNav、CityNavAgent、STMR 的官方记录已核验，见 `reference_audit.md`。它们分别提供基准、地理/城市任务、分层语义规划与记忆、空间表示。本文的差异必须落在灾情证据与观测预算上，不能落在“第一次把语言、地图、记忆接入无人机”。

**AerialClaw。** Ke Li 等，2026，*AerialClaw: An Open-Source Framework for LLM-Driven Autonomous Aerial Agents*，arXiv:2606.12142。官方摘要已明确提出技能、记忆和反馈闭环。因此普通 LLM→工具→UAV→反馈架构本身不够成为区别。项目内笔记把它称直接基座；正式稿需核查具体代码继承及许可证，不凭名字近似下结论。[官方记录](https://arxiv.org/abs/2606.12142)

**RescueADI。** Zhuoran Liu、Danpei Zhao、Bo Yuan，2024 arXiv:2410.13384，*RescueADI: Adaptive Disaster Interpretation in Remote Sensing Images with Autonomous Agents*。其任务已涵盖灾情工具规划、计数、面积与路径分析；不能把“灾害图像上的多步工具智能体”写成本稿首创。本文有条件的区别是通过运动改变观测证据及其任务成本，本轮已补读作者版任务、方法和实验正文，比较范围见 literature_topic_coverage.md。旧笔记的 TGRS 2025 出版说法仍待 publisher 核验。[官方记录](https://arxiv.org/abs/2410.13384)

**OpenFly。** 最新官方 v7（2026-03-01）的题名为 *Openfly: A comprehensive platform for aerial vision-language navigation*，arXiv:2502.18041v7，官方记录注明 accepted by ICLR 2026。本稿采用可核查的 v7 arXiv 元数据，末尾作者顺序也与 v7 对齐；不混用旧题名 *A Versatile Toolchain and Large-scale Benchmark...*。已检索官方摘要：包含多场景、多高度轨迹与关键帧模型。本文不能笼统说已有航空基准不考虑高度变化。[官方记录](https://arxiv.org/abs/2502.18041v7)

**AutoFly。** *AutoFly: Vision-Language-Action Model for UAV Autonomous Navigation in the Wild*，2026；arXiv:2602.09657，另已发现标明 ICLR 2026 的 OpenReview 官方 PDF。其连续自主导航/避障与本文理想化传感器动作不同，不能混合比较成功率。[官方论文](https://openreview.net/pdf?id=88RKxlFUNY)

**任务/运动规划中的信息获取。** Aidan Curtis 等，*Partially Observable Task and Motion Planning with Uncertainty and Risk Awareness*，RSS 2024，DOI 10.15607/RSS.2024.XX.118。其研究已把信息获取与风险纳入部分可观测规划，约束本文对“新闭环/新不确定性规划”的宽泛声明；直接移植到本文二维影像环境需另作适配论证。[RSS 官方记录](https://www.roboticsproceedings.org/rss20/p118.html)

**选择性预测。** Yonatan Geifman、Ran El-Yaniv 的 SelectiveNet（ICML 2019，PMLR 97:2151–2159）明确研究风险与覆盖率取舍。本文若声称可靠拒答，就必须测相应终点，不能用均值置信替代。[官方记录](https://proceedings.mlr.press/v97/geifman19a.html)

## CJA 写作与航空定位样例

本地样例 *Enhancing robustness of LLM-driven multi-agent systems through randomized smoothing* 已匹配 [CJA 出版社记录](https://www.sciencedirect.com/science/article/pii/S1000936125003851)：正式卷年为 2026（39(7), 103779），不是单看在线日期得到的 2025。可借鉴“明确问题—机制—仿真终点”的组织方式，不能照抄句子或将其安全证明移入本文。

另检索到 CJA 新近 *From aerial transportation to aerial interaction: A review* 的[出版社页面](https://www.sciencedirect.com/science/article/pii/S1000936126003808)，但正文抓取失败，暂只列候选，不能据搜索摘要写完整技术比较。

## 主题覆盖安排

按用户给定 22 主题分组推进，避免同义词重复堆文献：

1. Embodied intelligence / Embodied AI：用具体观测—动作定义；已有经典 VLN 与具身规划锚点，仍补综合概念的可靠来源。
2. UAV embodied intelligence / autonomous aerial vehicles / UAV agents：已有 AerialClaw、航空基准与 CJA 候选。
3. LLM-based agents / LLM for UAV / VLM for UAV / foundation models for aerospace autonomy：已有 LM-Nav、CityNavAgent、STMR、AerialClaw；继续精读航空基础模型近作。
4. VLN / aerial VLN / autonomous navigation：已有 AerialVLN、CityNav、OpenFly、AutoFly；需完成最相关方法全文比较。
5. VLA：已发现 AutoFly、OpenFly-Agent、FLIGHT，记录版本与出版状态后再筛选。
6. Task planning / hierarchical planning / tool-using agents：已有 CityNavAgent、RescueADI、TAMPURA；区分空间动作与静态图像工具。
7. Multimodal reasoning / spatial reasoning：已有 STMR 与相关航空工作；不得将仿真地理真值坐标当成模型自行学到的定位能力。
8. UAV disaster response / remote sensing agents：已有 RescueADI、ESARBench、xBD、RescueNet；重点查动作获取新证据与单幅解译的差异。
9. Closed-loop perception–planning–action / online replanning：已有 AerialClaw、主动感知、TAMPURA；还需对预算观测选择进行更窄检索。

候选发现不等于精读完成。最终 bibliography 只保留确实支持具体段落、元数据已核验的条目；不为凑 22 主题机械加入无关论文。

## 下一轮精读要提取的项目

对每个最相关方法记录任务、观测模态、动作空间、是否真实新观测、预算/成本、训练曝光、基线与终点。优先寻找可能推翻本稿 novelty 的近邻。研究缺口表述要有明确比较对象和适用范围，暂不使用 first、first-ever 或 no previous work。


后续结果：22 主题的定向查询已完成并记录于 [覆盖记录](literature_topic_coverage.md)。新增 CityEQA 直接先例，已修改正文及参考文献。以上“安排/下一轮”保留为历史计划，不能读作当前均未执行。
