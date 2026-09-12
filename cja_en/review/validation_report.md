# 最终审阅稿技术验证

日期：2026-09-11。对象：`cja_en/main.pdf`，30 页，18 条参考文献、4 幅正文图、7 张表（正文 4 张、附录 3 张）。

**技术一致性检查通过；当前仍是作者审阅稿。** 这不是独立专家外审结论或 CJA 格式认证。

## 实际检查与结果

- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` 编译成功。
- 编译错误、未解析引用、未解析标签、重复标签、缺失输入文件：0。
- LaTeX warning、Overfull 与 Underfull：0。
- 18 个引用键均有 BibTeX 条目；英文 TeX 无汉字，PDF 无 `??` 或重复 Appendix 前缀。
- 166 个受保护原论文/源码文件和 12 个固定运行输入仍由机器检查核对指纹。
- Figure 1--3 为 TikZ 矢量图；Figure 4 当前显示两个有意保留的截图占位框，固定 PNG 到位后自动替换。
- 通过 Poppler 以 120 dpi 渲染 1--30 页并逐页检查。正文、方程、四图、七表、附录与参考文献未见裁剪、重叠、缺字或缺图；Figure 1 的最终日志连线另行重新渲染检查。
- `visual_qa_matches_current_pdf = true`。

当前 PDF SHA-256：`e863ab49c8f8170a7136fcab48bc57bfc3818167b3ac051a769daad4e5f13c20`。

## 数值与语义边界

离线 7×5 分配曲线和三档 FOV 指标来自既有重建；160 题近匹配与固定视图控制按作者提供汇总呈现。200 指令 VLN B0--B3 表按作者确认直接采用 `paper_cja_v1/x6_main` 归档结果，没有在本轮复算。VLN 表仅作组件接入诊断，不支持模块增益或跨种子结论。

VLM 身份按作者确认为本地 Qwen2.5-VL-7B-Instruct。该确认不替代 checkpoint revision、prompt hash、dirty 源码快照和训练曝光归档。Figure 4 截图只说明平台功能。

如替换 Figure 4 PNG 或修改 TeX，应重新编译并更新视觉 QA 的 PDF hash。
