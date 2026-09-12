# 投稿前检查清单

状态：**英文审阅稿已纳入补实验并完成平台主线重构；未达到可直接投稿状态。** 2026-09-08。技术可编译与科学投稿成熟度分别判断；内部复审仍为 Major revision。

## 已形成的稿件内容

- [x] Research Article 形式：一个明确问题、八章正文，平台、智能体、协议、结果、局限和结论分章。
- [x] 标题已从候选中结合实际证据收敛；有摘要和六个关键词。
- [x] 贡献与证据对齐；无 97%/3% 归因、伪可识别性定理、虚构显著收益和泛化保证。
- [x] 三幅正文矢量图、六张正文表；无缺图或伪现场图。离线程序生成资产具有输入指纹，补实验表来自作者提供的结果摘要。
- [x] 作者、单位、通讯作者、资助、CRediT、利益冲突、致谢、数据可用性以真实待确认状态保留，没有编造。
- [x] AI 使用说明作为作者待审文本提供，不预称所有作者已核验。
- [x] 最终内部复审已保存，并至少落实一轮实质修订。
- [x] LaTeX 已实际编译并进行逐页视觉检查；最终技术结果见 validation_report。

## 科学证据必须补齐

- [x] VLM 家族、版本与规模已由作者确认为本地 `Qwen/Qwen2.5-VL-7B-Instruct`；答题接口 temperature=0.1、max tokens=300。
- [ ] Qwen checkpoint revision、有效 top-p/repetition penalty、prompt hash 与模型文件 hash。
- [ ] dirty runtime 源码归档；当前代码与运行代码差异。
- [ ] 冻结题集、图像/ROI manifest、评分规则、selection/final 关系核查。
- [ ] 权重 hash、训练事件/样本实体及严格未见事件 checkpoint 的复核。
- [x] 已在新目录完成 160 题匹配预算重跑和每题 5 次原图不动重问；论文已纳入自洽端点。
- [ ] 将新运行的逐题 episodes、paired tests、summary 和 manifest 复制到当前工程并纳入 SHA-256/算术审计；当前两份摘要有两处已记录的计数矛盾。
- [ ] 完整移动策略的多次生成重复；现有不动重问只估计固定输入的生成波动。
- [ ] 若强化算法贡献，实际验证任务条件化观测策略，而不是只在结论提出。
- [ ] 若强化可靠性贡献，独立开发阈值与 held-out risk–coverage，而不是把事后曲线写成保证。
- [ ] 原始来源影像的合法可分享配对案例及精确输入核验；现稿没有伪造现场截图。
- [ ] 平台像素—地理变换与灾前/灾后窗口一致性、固定动作回放、渲染/感知/Qwen/整步时延以及接口覆盖的归档结果。
- [ ] 可核验的 DisasterClaw 控制台和地理场景截图；正文当前只有实现对应的矢量示意图，不将其当作真实运行证据。

## 文献与期刊要求

- [x] 新稿 18 个引用键均有条目，核心元数据已对齐其官方来源；新增近邻的来源与作用见 reference_audit。
- [x] 22 个主题定向检索及 CityEQA/AerialClaw/RescueADI 作者版相关正文比较已记录；仍非穷尽性系统综述，正式出版版逐页阅读范围有限。
- [ ] RSS/NeurIPS 会议页码等少量元数据终核；现存三个 BibTeX 缺页提示不通过虚填消除。
- [ ] CJA 最新 Guide for Authors 的专用样式、摘要/关键词限制、参考文献截断规则、highlights 要求与 cover letter 要求终核。ScienceDirect 正文访问受限；没有让第三方模板冒充官方依据。
- [x] 使用用户允许的 Elsevier `elsarticle` 后备方案；BibTeX 数字引用。官方通用 LaTeX 说明支持该类模板：[Elsevier LaTeX instructions](https://www.elsevier.com/researcher/author/policies-and-guidelines/latex-instructions)。
- [x] special issue 内容适配依据作者提供的征稿正文评估：主适配 intelligent perception / situational understanding；不宣称飞行安全认证或群体智能。
- [ ] 投稿系统专栏选项 **Embodied AI for Aerospace Vehicles**、Title Page 模板、补充材料与开放数据要求在正式提交时确认。
- [ ] 作者最终确认全部署名、单位/邮箱、基金、声明、原创及未一稿多投状态；未替作者作出这些事实声明。

作者提供的征稿截止为 **2026-12-31**，出版年份为 **2028**。这两个日期仅按提供正文记录，尚未完成当前在线投稿系统的独立确认。没有执行投稿、邮件或外部发布。
