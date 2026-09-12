# Figure 4 控制台截图指南

论文已经为 Figure 4 预留两个截图位置。截图用于说明 DisasterClaw 的功能界面，不作为实验定量证据。请保持两张图使用同一个浏览器窗口尺寸、缩放比例和界面主题。

## 1. 启动平台

在服务器或本机项目目录分别启动后端和前端：

```bash
conda activate disasterclaw
./scripts/run_backend.sh
```

```bash
./scripts/run_frontend.sh
```

本机开发模式打开 `http://127.0.0.1:5173`。如果使用服务器构建版，请按项目根目录 `README.md` 的 SSH 隧道方法访问 `http://127.0.0.1:5011`。

建议使用 1920×1080 或 2560×1440 的浏览器窗口、90%–100% 页面缩放。隐藏书签栏和开发者工具。截图前确认页首显示 **Socket Connected** 和 **UAV Hovering**，并检查页面中没有 `.env` 内容、API 密钥、服务器路径或个人信息。

## 2. 截图 (a)：平台总览

1. 在 Situation Map 选择一个建筑和损毁标注较丰富的 post-disaster 场景，例如 Hurricane Michael。
2. 保留灾后影像叠加层、彩色建筑轮廓、UAV 标记和地图比例尺。
3. 调整页面滚动位置，使 Situation Map、UAV/Robot 状态、Task、Perception 和 Mission Log 等主要区域尽量同时可见。
4. 可以执行一次 `detect_disaster` 或普通检查任务，使 Perception 面板显示真实输出；不要为了截图手工伪造重观测事件。
5. 保存为：

   `cja_en/figures/console/console_overview.png`

这张图重点表现“操作员—地图—无人机—任务—感知”的平台组织关系。

## 3. 截图 (b)：闭环智能体状态

1. 保持与截图 (a) 相同的场景和浏览器尺寸。
2. 选择 **AI Auto**，在 Task 面板提交一条检查或导航任务。可使用“最近的完全损毁建筑位于无人机哪个方向？”；若当前数据不适合该问题，则使用能在该场景产生有效目标的既有 VLN 指令。
3. 点击 **Run AI Mission**，等待至少 2–3 个执行步骤或任务自然结束。
4. 截图中保留地图轨迹或位置变化，并让 Perception、智能体决策/查询、剩余预算和 Mission Log 可读。若该任务没有触发重观测，保留真实的搜索、移动或回答过程即可。
5. 保存为：

   `cja_en/figures/console/console_agent_loop.png`

这张图重点表现“观测—证据—决策—动作—日志”的闭环。

## 4. 截图与编译

macOS 可用 `Command + Shift + 4` 后按空格选择浏览器窗口；也可在 Chrome 开发者工具中按 `Command + Shift + P`，运行 **Capture screenshot**。优先截取当前可见窗口，避免 full-page 截图把固定布局拉成长图。

两张 PNG 放入上述路径后运行：

```bash
make -C cja_en pdf
```

LaTeX 会自动用真实截图替换占位框。不要改变文件名；若必须裁剪，须保留地图、任务/智能体状态和日志之间的界面关系。
