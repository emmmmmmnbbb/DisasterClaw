# 最终审阅稿技术验证

日期：2026-09-06。对象：`cja_en/main.pdf`，23 页，18 条参考文献、3 幅矢量图、6 张表。

**技术一致性检查通过；科学投稿成熟度仍为 Major revision。** 这不是 ARS 总体 integrity PASS、独立专家外审结论或 CJA 格式认证。

## 实际检查与结果

执行 `make -C cja_en pdf`，由 latexmk、pdfLaTeX 与 BibTeX 编译。执行 `python3 cja_en/scripts/validate_manuscript.py`，读取 PDF 和编译日志并核验输入。

- 编译错误、未解析引用、未解析标签、重复标签、缺失输入文件：0。
- LaTeX warning、Overfull 与 Underfull：0。
- 18 个引用键均有 BibTeX 条目，无未使用条目；英文源文件未检出汉字，PDF 无 `??` 或重复 Appendix 前缀。
- 166 个受保护原论文/源码文件、12 个固定原始运行输入：SHA-256 均与此前清单一致。
- 分析文件绑定当前审计 hash；归档算术核验无不一致，在线记录无重复 configuration/question 键。
- 通过 Poppler 将当前 PDF 的 1–23 页全部渲染为 PNG，逐页实际查看。正文、方程、三图、六表、附录与参考文献未见裁剪、重叠、缺字或缺图。作者查询与实验 TODO 为审阅稿的有意保留内容。
- `visual_qa_matches_current_pdf = true`；视觉记录严格绑定此次 PDF，不沿用旧版检查结论。

当前 PDF SHA-256：`158516818c3e8d9c870df0894a87a9539900b21cb1cdf28342f71315fdd66742`。

## 保留的非致命提示

BibTeX 有三条缺页码提示：`curtis2024tampura`、`ovadia2019can`、`romano2020adaptive`。相关论文存在性和核心元数据已核验；尚未直接核实的页范围没有虚填。RSS 条目已给官方 DOI。这些提示不妨碍审阅稿编译，但仍列入参考元数据终核。

## 数值与语义验证边界

离线 7×5 分配曲线和三档 FOV 主指标由原始逐建筑预测重建，容差 1e-9。在线 1,600 条记录对应 160 道题在 10 个配置下的重复评估，不能当 1,600 个独立样本。新增 ROI bootstrap 与共同匹配敏感性不调用模型，结果条件化于已记录预测及拟合配置。

该验证不认证原始图像标签、实际 VLM 身份、dirty 运行源码、训练曝光声明或评分器的端到端正确性；原始实体仍有缺项。事后风险曲线也不是可部署的可靠拒答保证。

## 可复查材料

- [机器检查记录](technical_validation.json)
- [逐页视觉检查记录](visual_qa.json)
- [原始结果审计](downloaded_results_audit.json)
- [新增分析](manuscript_analysis.json)
- [内部复审与落实](final_reviewer_simulation.md)
- [任务完成情况及后续依赖](task_completion_status.md)

如修改 TeX、图表或重新生成 PDF，应重新编译并更新逐页检查，旧视觉 hash 不再覆盖新版。
