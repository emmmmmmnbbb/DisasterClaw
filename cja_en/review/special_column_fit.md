# 专栏适配分析

日期：2026-09-05。征稿依据：作者本会话粘贴的完整通知；微信链接无法可靠提取，未猜测缺失正文。

## 通知信息

- 专栏：**Embodied AI for Aerospace Vehicles（空天飞行器具身智能）**。
- 截稿：2026-12-31；计划出版年份：2028，按作者所贴通知记录。实际投稿前需再次核实。
- 稿件：原创 Research Article、Review、Case Study；本项目按完整 Research Article 准备。
- 通知所列 Guest Editors：Qinglei Hu、Xiaodong Shao、Zheng H. (George) Zhu、Youmin Zhang。在线征稿索引与通知人员条目可能更新，最终以投稿系统为准。
- 范围包括智能感知与态势理解、非结构化环境控制、博弈决策、群体智能、基础模型策略生成、机载推理与能耗、可信具身智能、故障恢复和空天机器人学习/操作。
- 提交完整论文；不得重复发表或同时投其他期刊；在系统选择该专栏名称。通知给出 [CJA 主页](https://www.sciencedirect.com/journal/chinese-journal-of-aeronautics) 和 [Title-Page 页面](http://hkxb.buaa.edu.cn/CN/column86.shtml)。

## 本文最合适的定位

**主要匹配 Intelligent perception & situational understanding for aerospace autonomy。** 次要关联运行中不确定性评估；目前尚不支持安全认证、可信性保证、机载轻量化、群体协同或自进化恢复。

任务背景应是单 UAV 灾后观测与判读：改变地理位置和高度，会同时改变视场覆盖与有效细节；额外观测占用任务预算，只有提高任务相关信息质量才值得执行。这一取舍应贯穿问题定义、动作设计和实验，而非只在引言出现 UAV 一词。

## 闭环是否真实存在

当前实现链条：

1. 环境：带地理参考的灾前/灾后影像及其重渲染观测。
2. 状态与观测：单 UAV 的位置/高度、当前视图、感知类别/概率、语义地图和预算。
3. 认知：解析语言目标，利用感知与空间表示组织任务相关证据。
4. 决策：HSPM/VLN 搜索或 RecheckController 选择继续、居中、下降、回答/终止。
5. 行动：运动适配器更新位置或高度。
6. 反馈：在新状态渲染视图并重新感知，后续步骤据新证据决策。

证据入口：`backend/app.py` 的导航及 VQA 循环、`hspm_planner.py`、`recheck.py`、`agent_vqa.py`、`perception.py`、`world.py` 与 `mock_adapter.py`。

因此本文不只是一次 Image→Model→Text；但离线建筑重采样实验只能验证观测机制的一部分，必须有对应在线日志说明动作确实发生、观测确实更新、决策确实消费新观测。

## 必须公开的范围

- 这是理想化单 UAV 传感器—运动仿真，不是多无人机集群。
- 当前代码允许静止悬停/直接位移，没有固定翼飞行动力学和航迹可行性约束；不能仅凭应用设想称完成大型固定翼验证。
- 卫星影像裁剪/缩放不产生源图之外的新细节，不提供真实低空视差、遮挡变化或动态环境验证。
- 高度、FOV 和 GSD 是仿真观测假设；对 xBD 统一采用源 GSD 时要说明其设定和实际元数据核对状态。
- 动作次数预算可以报告；不能直接等同真实航时、能耗或安全代价。

## 当前格式核验状态

[CJA Guide for Authors](https://www.sciencedirect.com/journal/chinese-journal-of-aeronautics/publish/guide-for-authors) 的自动抓取受访问限制；尚未据当前正文核验字数、Highlights、参考文献样式和模板细则。第三方博客不作为最终依据。

本地 `MinerU_markdown_写作模版_2092533610450280448.md` 是另一篇关于 LLM 多智能体随机平滑的文章抽取文本，只可参考组织方式。其多智能体、安全证明和实验不属于本项目，不能移植为本文贡献，也不能把 OCR 版式当作 CJA 投稿规范。
