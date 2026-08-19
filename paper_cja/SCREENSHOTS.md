# 论文截图拍摄说明

把 PNG 放到 `paper_cja/figures/`，文件名与正文 `\includegraphics` 一致。
建议分辨率：全界面 1920×1080 或更高，地图局部 1600×900；导出前关掉浏览器书签栏。
论文编译时若文件不存在，会显示灰色占位框，补图后重新 `latexmk` 即可。

启动（仓库根目录）：

```bash
# 终端 1
cd backend && python app.py
# 终端 2
cd frontend && npm run dev
```

打开前端地址（默认 `http://127.0.0.1:5173`）。确认顶栏 Socket Connected、UAV Hovering。

## 图 ui_overview.png — 全界面总览

对应正文 `fig:ui-overview`。

1. 在地图左侧灾害列表选一个**灾后瓦片丰富**的事件（推荐 `hurricane-michael` 或 `hurricane-florence`）。
2. 点选一块 POST 瓦片，等卫星叠加层与建筑足迹加载完。
3. 打开图层：底图（Esri/OSM）、xBD 叠加、建筑标注、足迹。
4. 不要发任务。画面应同时看到：
   - 顶栏 DisasterClaw / Manual·AI Auto
   - 左栏 UAV 位姿（经纬高、航向、电池）
   - 中栏态势地图（瓦片 + 无人机图标）
   - 右栏 Task Console（AI Task 与 VLN 两个文本框）
   - 底栏 Perception（尚空）与 Mission Log
5. 全窗口截图，不要裁掉底栏。

## 图 map_overlay.png — 地理配准与标注

对应正文 `fig:ui-map`。

1. 在上一状态把地图放大到单块瓦片几乎铺满中栏。
2. 确认能看清：彩色建筑多边形（绿=完好、黄=轻微、橙=严重、红=损毁）、瓦片边界、UAV 图标。
3. 若有历史轨迹，先飞一小段再截（见下一条），使轨迹折线可见。
4. 只截中栏地图，或全界面但地图占主体。

## 图 inspect_perception.png — 点选巡检与三视图感知

对应正文 `fig:ui-perception`。

1. 模式切到 Manual。地图上点一栋**红色损毁建筑**附近。
2. 点 Inspect / 受灾检测（会自动构造 AI 任务：飞到该点 30 m 悬停并检测）。
3. 等待底栏 Perception 出现图像。
4. 分别点三个页签截三张，再拼成横排（推荐）：
   - `UAV 视场`：正射裁块
   - `YOLO 标注`：检测框
   - `SegFormer`：分割叠加
5. 同一屏里尽量露出：风险徽章、受损/完好/车辆计数、Qwen-VL 结论（若已加载本地 VLM）。
6. 拼图可用任意工具横排，最终保存为 `inspect_perception.png`。

没有本地 VLM 时不要空等：截 YOLO+分割即可，图注写明「视觉语言模型摘要可选」。

## 图 vln_closedloop.png — 语言导航闭环

对应正文 `fig:ui-vln`。

1. 模式切到 AI Auto。
2. VLN 框输入：`飞到北侧寻找完全损毁的建筑。`，点 Run VLN。
3. 等无人机走出至少 3 步，Perception 出现 `VLN 语言导航` 蓝条。
4. 截全界面，必须同时看到：
   - 地图上 UAV 在动、轨迹折线变长
   - 底栏指令原文、当前技能（飞行/悬停/复核）、命中或搜索中
   - 语义地图统计：步、已探索格、地标、候选
   - 若触发复核：不确定性进度条
5. 任务未结束时截，不要等 Stop。

## 图 ai_plan.png — 一次性任务规划

对应正文 `fig:ui-plan`。

1. 右栏 AI Task 用默认或：`飞到当前地图中心附近进行低空观察，并给出简短态势汇报。`
2. 点 Run AI Mission，等右侧出现分步计划（fly_to_geo / hover / detect_disaster / report_observation）。
3. 截右栏计划列表 + 底栏日志正在执行的那几行。

## 图 recheck_descend.png — 下降居中（可选，强烈建议）

对应正文 `fig:ui-recheck`。

较难拍到（端到端里复检很少触发）。两种做法：

- **演示**：语言导航进行中，左栏 Altitude 从约 30 m 降到 20 m，同时视场明显变小。左右并排：巡航视场 | 降后视场，并在图上标高度。
- **评测回放**：若有 `runs/benchmarks/.../episodes.jsonl` 里 `recheck_triggered=true` 的回合，用该瓦片手工飞到同一点，先 30 m 截视场，再手动相对下降 10 m 再截。

没有可靠画面就先不放这张，正文占位框会提示待补。

## 文件名对照

| 文件 | 正文标签 | 优先级 |
|------|----------|--------|
| `figures/ui_overview.png` | `fig:ui-overview` | 必补 |
| `figures/map_overlay.png` | `fig:ui-map` | 必补 |
| `figures/inspect_perception.png` | `fig:ui-perception` | 必补 |
| `figures/vln_closedloop.png` | `fig:ui-vln` | 必补 |
| `figures/ai_plan.png` | `fig:ui-plan` | 建议 |
| `figures/recheck_descend.png` | `fig:ui-recheck` | 有则加 |

不要在截图里出现绝对路径、`.env`、API key、个人信息。建筑足迹是公开 xBD 标注，可以入文。
