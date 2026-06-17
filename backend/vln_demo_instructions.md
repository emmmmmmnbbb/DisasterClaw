# VLN 语言目标导航 — PoC 演示脚本

本文件给出在 DisasterClaw 里跑通 BEV 语言目标导航闭环（`vln_navigator.py` +
`run_vln_episode`）的演示步骤与几条手写指令。

## 定位

- 这是 CityNav 式的"地理 BEV 上的语言指代导航"，不是第一视角 AVLN。
- 每步：`perceive_at` 裁俯视视场 → YOLO 检测做目标 grounding → 朝目标质心步进 →
  目标进入视场中心半径（默认 35m）即判定到达，或步数预算（默认 12）耗尽结束。

## 启动

```bash
# 后端（自带 YOLO + SegFormer 感知）
bash scripts/run_backend.sh        # 默认 127.0.0.1:5011

# 前端
bash scripts/run_frontend.sh
```

在前端右侧 Task Console 的 **"VLN 语言目标导航"** 卡片里输入指令，点 **Run VLN**。
执行过程中：

- 地图上无人机会逐步移动（`world_state` 流式更新，可观察轨迹）。
- Perception 面板每步刷新当前俯视 patch 与检测结果。
- Log 面板按 `[VLN k/N]` 打印每步的 grounding / 决策思考。
- 结束后 Mission Report 给出到达与否的摘要。

## 推荐瓦片

默认初始瓦片为 `palu-tsunami_00000118_post_disaster`（~1540 完全损毁，1024×1024
卫星图），目标密度高，适合演示 grounding 与到达。
也可用 RescueNet 模式：`DATASET_MODE=rescuenet`（`rescuenet_12215_post_disaster`，
4000×3000 无人机高清，含 destroyed / major 建筑与车辆）。

> 注意：导航全程需 UAV 落在 POST 灾后瓦片覆盖范围内；否则该步会判 "无 POST 覆盖"
> 并中止（与 `detect_disaster` 的约束一致）。

## 手写指令（覆盖目标 + 方向先验）

1. `飞到北侧寻找完全损毁的建筑。`
   - 目标类别 → {完全损毁建筑}；方向先验 → 北。先沿北探索，命中后朝质心收敛。
2. `在附近找受损建筑并飞过去。`
   - 目标类别 → {轻微/严重/完全损毁建筑}；无方向 → 方形螺旋搜索。
3. `向东去查看有没有车辆。`
   - 目标类别 → {车辆}；方向先验 → 东。
4. `寻找积水/洪水区域。`
   - 目标类别 → {水池/积水区域}；无方向 → 螺旋搜索。
5. `飞到西南方向的严重损伤建筑。`
   - 目标类别 → {严重损伤建筑}；方向先验 → 西南。

## 可调参数（环境变量）

| 变量 | 默认 | 含义 |
| --- | --- | --- |
| `VLN_STEP_BUDGET` | 12 | 最大决策步数 |
| `VLN_ARRIVAL_RADIUS_M` | 35 | 目标质心进入此半径视为到达 |
| `VLN_MAX_STEP_M` | 80 | 单步朝目标移动上限（防越界/跳瓦片） |
| `VLN_EXPLORE_STEP_M` | 90 | 未命中时的搜索步长 |
| `VLN_USE_LLM_STOP` | 0 | 到达候选时是否让 planner LLM 复核（控时延，默认关） |

## 已知局限（PoC）

- grounding 用 YOLO 类别质心；退化视场（裁到整图/贴边 clamp）当作未命中，仅探索。
- 仅在单个 POST 瓦片内演示；跨瓦片长距离导航依赖 `fly_to_geo` 的自动瓦片对齐，未做专门优化。
- 开放词汇 grounding（LocateAnything）为二期接入点（`VlnNavigator.locate_ground_fn`），
  当前未启用。
