# 导出素材

原始文件：`pre_disaster.png`、`post_disaster.png`、`*_disaster_label.json`，逐字节复制自本机 xBD 数据集，同一 tile `nepal-flooding_00000484`。

派生图：`building_overlay.png` 由原始 post 图和 polygon 标签绘制；四类 `*_example.png` 由同一真实 Benchmark 题目的 ROI、target 与建筑标注绘制。完整题目和答案见 `benchmark_examples.json`。图中的青色框为 ROI，紫色点为 target。

`perception_*` 是平台已有 UAV 视场输出的原样副本，未核实是否对应上述 tile。

`map_anchored_scene.png` 如存在，是从本地 Esri World Imagery 缓存拼接，地理配准叠加真实 xBD tile，并标出 tile 边界、ROI、UAV 起点的派生图，不是浏览器截图。`map_layers.json` 保存图层坐标。

仓库中未找到现成的高清控制台、AI Auto/VLN、mission log 截图；因此没有用伪造或其他图替代。
