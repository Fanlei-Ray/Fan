import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const REPORT = "E:\\FL_Personal\\openarm_mujoco-master\\openarm_mujoco-master\\upper_computer\\report";
const BUILD = path.join(REPORT, "build");
const ASSETS = path.join(REPORT, "assets");
const FINAL = path.join(REPORT, "OpenArm_上位机课程项目汇报.pptx");
const RENDERED = path.join(BUILD, "rendered");

const C = {
  canvas: "#FFFFFF",
  ink: "#111111",
  muted: "#5B6470",
  panel: "#EDEDED",
  panel2: "#F6F7F8",
  rule: "#B8BCC4",
  blue: "#3D8DFF",
  blueLight: "#D0EDFA",
  orange: "#F28C28",
  orangeLight: "#FCE6CC",
  red: "#C73737",
  redLight: "#FBE1E1",
  green: "#198754",
};
const FONT = "Microsoft YaHei";

async function imageBytes(name) {
  const bytes = await fs.readFile(path.join(ASSETS, name));
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function addShape(slide, { name, geometry = "rect", left, top, width, height, fill = "none", lineFill = "none", lineWidth = 0, radius }) {
  return slide.shapes.add({
    geometry,
    name,
    position: { left, top, width, height },
    fill,
    line: { style: "solid", fill: lineFill, width: lineWidth },
    ...(radius ? { borderRadius: radius } : {}),
  });
}

function addText(slide, text, { name, left, top, width, height, fontSize = 24, color = C.ink, bold = false, align = "left", valign = "top", fill = "none", lineFill = "none", lineWidth = 0, margins = 0 }) {
  const box = slide.shapes.add({
    geometry: "textbox",
    name,
    position: { left, top, width, height },
    fill,
    line: { style: "solid", fill: lineFill, width: lineWidth },
  });
  box.text = text;
  box.text.style = {
    fontSize,
    typeface: FONT,
    color,
    bold,
    alignment: align,
    verticalAlignment: valign,
    autoFit: "shrinkText",
    insets: { top: margins, right: margins, bottom: margins, left: margins },
  };
  return box;
}

function addHeader(slide, title, page, kicker = "OPENARM / UPPER COMPUTER") {
  addText(slide, kicker, { name: `kicker-${page}`, left: 54, top: 28, width: 420, height: 26, fontSize: 16, color: C.muted, bold: true });
  addText(slide, title, { name: `title-${page}`, left: 54, top: 65, width: 1165, height: 68, fontSize: 48, bold: true });
  addShape(slide, { name: `rule-${page}`, geometry: "straightConnector1", left: 54, top: 145, width: 1172, height: 0, lineFill: C.ink, lineWidth: 1 });
  addText(slide, String(page).padStart(2, "0"), { name: `page-${page}`, left: 1170, top: 674, width: 56, height: 24, fontSize: 15, color: C.muted, align: "right" });
}

function addBulletList(slide, items, { left, top, width, fontSize = 24, gap = 18, accent = C.blue, lineHeight = 46, name = "bullets" }) {
  let y = top;
  items.forEach((item, index) => {
    addText(slide, "•", { name: `${name}-dot-${index}`, left, top: y - 1, width: 24, height: 32, fontSize: fontSize + 2, color: accent, bold: true });
    addText(slide, item, { name: `${name}-text-${index}`, left: left + 30, top: y, width: width - 30, height: lineHeight, fontSize, color: C.ink });
    y += lineHeight + gap;
  });
}

function setNotes(slide, body, sources) {
  slide.speakerNotes.textFrame.setText(`${body}\n\n[Sources]\n${sources.map((s) => `- ${s}`).join("\n")}`);
  slide.speakerNotes.setVisible(true);
}

function addMetric(slide, left, stat, label, detail, accent, name) {
  addShape(slide, { name: `${name}-panel`, geometry: "roundRect", left, top: 314, width: 356, height: 300, fill: C.panel2, lineFill: C.rule, lineWidth: 1, radius: "rounded-xl" });
  addShape(slide, { name: `${name}-accent`, left: left + 22, top: 340, width: 56, height: 7, fill: accent });
  addText(slide, stat, { name: `${name}-stat`, left: left + 22, top: 370, width: 310, height: 92, fontSize: 58, bold: true, color: accent });
  addText(slide, label, { name: `${name}-label`, left: left + 22, top: 469, width: 310, height: 40, fontSize: 27, bold: true });
  addText(slide, detail, { name: `${name}-detail`, left: left + 22, top: 525, width: 310, height: 62, fontSize: 19, color: C.muted });
}

async function build() {
  await fs.mkdir(RENDERED, { recursive: true });
  const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });

  // 01 — Cover, based on Codex Grid slide-08 image field.
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.canvas;
    addText(slide, "COURSE PROJECT / 课程项目阶段汇报", { name: "cover-kicker", left: 54, top: 44, width: 510, height: 28, fontSize: 17, color: C.blue, bold: true });
    addText(slide, "OpenArm\n视觉—上位机联调", { name: "cover-title", left: 54, top: 126, width: 595, height: 164, fontSize: 60, bold: true });
    addText(slide, "仿真软件闭环已完成\n真机控制保持安全闭锁", { name: "cover-subtitle", left: 58, top: 334, width: 510, height: 105, fontSize: 30, color: C.muted, bold: true });
    addShape(slide, { name: "cover-accent", left: 58, top: 481, width: 168, height: 8, fill: C.blue });
    addText(slide, "上位机工作内容 · 仿真演示 · 真机侧演示 · 部署阻塞分析", { name: "cover-footer", left: 58, top: 526, width: 550, height: 64, fontSize: 21, color: C.muted });
    addShape(slide, { name: "cover-image-field", geometry: "roundRect", left: 665, top: 36, width: 560, height: 632, fill: "#0B0B0B", radius: "rounded-xl" });
    slide.images.add({ blob: await imageBytes("openarm_model.png"), contentType: "image/png", alt: "OpenArm 双臂 MuJoCo 模型", fit: "contain", position: { left: 690, top: 58, width: 510, height: 584 } });
    setNotes(slide, "开场先给出阶段结论：本次负责的是上位机集成。仿真侧可以形成视觉—上位机—任务执行闭环；真机侧没有为了赶进度而绕过安全条件。", ["media/v2.png", "upper_computer/README.md"]);
  }

  // 02 — Responsibility scope.
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.canvas;
    addHeader(slide, "我的工作：把视觉、仿真与电控统一到一个上位机入口", 2);
    const cols = [54, 449, 844];
    const titles = ["界面与任务流程", "三类接口适配", "安全与可验证性"];
    const bodies = [
      "目标画面与检测框\n任务确认 / 回零 / 取消\n状态机与会话日志",
      "离线回放\nMuJoCo WebSocket\nRealSense / YOLO WebSocket\nUSART10 协议工具",
      "坐标系与工作空间校验\n超时、重连、队列限流\n急停锁存与执行闭锁\n自动测试和联调文档",
    ];
    cols.forEach((x, i) => {
      if (i > 0) addShape(slide, { name: `scope-vline-${i}`, geometry: "straightConnector1", left: x - 28, top: 205, width: 0, height: 348, lineFill: C.rule, lineWidth: 1 });
      addText(slide, `0${i + 1}`, { name: `scope-num-${i}`, left: x, top: 194, width: 70, height: 44, fontSize: 28, color: C.blue, bold: true });
      addText(slide, titles[i], { name: `scope-title-${i}`, left: x, top: 254, width: 330, height: 47, fontSize: 31, bold: true });
      addText(slide, bodies[i], { name: `scope-body-${i}`, left: x, top: 332, width: 330, height: 225, fontSize: 23, color: C.muted });
    });
    addText(slide, "核心原则：UI 不直接绑定某一个视觉程序或电机协议；通过传输层切换数据源，通过安全层决定是否允许执行。", { name: "scope-conclusion", left: 54, top: 600, width: 1120, height: 48, fontSize: 24, bold: true, color: C.ink });
    setNotes(slide, "这一页说明上位机工作不是单一界面，而是把视觉、电控、仿真和安全状态整合为可替换的模块。", ["upper_computer/PROJECT_STRUCTURE_ZH.md", "upper_computer/src/openarm_upper/"]);
  }

  // 03 — Workload metrics, based on Codex Grid slide-19.
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.canvas;
    addHeader(slide, "当前上位机交付规模已经覆盖开发、测试与联调资料", 3);
    addText(slide, "统计不含 vendor 原始资料、运行日志、.venv 与缓存；用于展示当前交付模块规模，不把同学提供的代码计入个人新增代码。", { name: "metrics-context", left: 54, top: 174, width: 1150, height: 58, fontSize: 22, color: C.muted });
    addMetric(slide, 54, "18 / 1,754", "应用源码文件 / 行", "GUI、消息模型、安全、状态机、日志、传输与协议", C.blue, "metric-src");
    addMetric(slide, 462, "28", "自动测试", "9 个测试文件，覆盖消息、WebSocket、安全和 USART10", C.green, "metric-test");
    addMetric(slide, 870, "3 + 6 + 5", "模式 / 配置 / 启动入口", "回放、MuJoCo、真机视觉；统一课程演示菜单已整理", C.orange, "metric-run");
    addText(slide, "另有：3 个只读工具、5 份联调文档、2 个仿真视觉文件（1,009 行）", { name: "metrics-foot", left: 54, top: 646, width: 1050, height: 30, fontSize: 19, color: C.muted });
    setNotes(slide, "工作量用可复算的文件和行数表达。这里强调是当前上位机交付范围，不把 vendor 中电控或视觉同学的原始代码当作个人工作量。", ["upper_computer/report/build/source-notes.txt", "upper_computer/tests/"]);
  }

  // 04 — Architecture diagram. Connectors are deliberately created first.
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.canvas;
    addHeader(slide, "同一套 UI，通过传输适配器切换三种联调模式", 4);
    const y = 312;
    // connectors first, behind nodes
    addShape(slide, { name: "arch-arrow-1", geometry: "rightArrow", left: 286, top: y + 38, width: 74, height: 34, fill: C.blueLight, lineFill: C.blue, lineWidth: 1 });
    addShape(slide, { name: "arch-arrow-2", geometry: "rightArrow", left: 570, top: y + 38, width: 74, height: 34, fill: C.blueLight, lineFill: C.blue, lineWidth: 1 });
    addShape(slide, { name: "arch-arrow-3", geometry: "rightArrow", left: 855, top: y + 38, width: 74, height: 34, fill: C.blueLight, lineFill: C.blue, lineWidth: 1 });
    const nodes = [
      { x: 54, w: 232, title: "数据源", body: "回放 JSONL\nMuJoCo RGB\nRealSense / YOLO", fill: C.panel2 },
      { x: 360, w: 210, title: "传输适配", body: "DetectionBatch\nTaskStatus\n有界队列 / 重连", fill: C.panel2 },
      { x: 644, w: 211, title: "上位机核心", body: "画面与目标\n安全校验\n任务状态机 / 日志", fill: C.blueLight },
      { x: 929, w: 297, title: "执行边界", body: "MuJoCo：允许仿真执行\n真机视觉：只观察\nUSART10：只读", fill: C.panel2 },
    ];
    nodes.forEach((n, i) => {
      addShape(slide, { name: `arch-node-${i}`, geometry: "roundRect", left: n.x, top: y, width: n.w, height: 214, fill: n.fill, lineFill: C.rule, lineWidth: 1, radius: "rounded-xl" });
      addText(slide, n.title, { name: `arch-node-title-${i}`, left: n.x + 22, top: y + 25, width: n.w - 44, height: 42, fontSize: 29, bold: true });
      addText(slide, n.body, { name: `arch-node-body-${i}`, left: n.x + 22, top: y + 92, width: n.w - 44, height: 98, fontSize: 21, color: C.muted });
    });
    addText(slide, "上位机不把“看见目标”直接等同于“允许运动”", { name: "arch-claim", left: 54, top: 190, width: 900, height: 52, fontSize: 32, bold: true });
    addText(slide, "执行权限由坐标、时效、工作空间、接口完备度和运行模式共同决定。", { name: "arch-subclaim", left: 54, top: 246, width: 1040, height: 36, fontSize: 22, color: C.muted });
    setNotes(slide, "重点讲右侧的执行边界：相同界面并不意味着相同权限。仿真模式能够执行，真机模式在证据不完整时必须被锁住。", ["upper_computer/src/openarm_upper/transports/", "upper_computer/src/openarm_upper/safety.py", "upper_computer/config/"]);
  }

  // 05 — Simulation demo image split, based on slide-08.
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.canvas;
    addHeader(slide, "演示一：仿真上位机展示完整的视觉—任务闭环", 5, "LIVE DEMO / MUJOCO FIRST");
    addText(slide, "启动顺序", { name: "sim-order-title", left: 54, top: 184, width: 235, height: 40, fontSize: 27, bold: true, color: C.orange });
    addText(slide, "① 运行 MuJoCo 视觉桥\n② 运行仿真上位机\n③ 识别橙色方块\n④ 确认执行抓取放置", { name: "sim-order", left: 54, top: 242, width: 465, height: 190, fontSize: 28, bold: true });
    addBulletList(slide, ["640×480 虚拟相机，5 FPS 持续中间帧", "检测框、置信度与 base_link 坐标", "PLANNING / EXECUTING / SUCCEEDED 状态回传"], { left: 54, top: 470, width: 520, fontSize: 21, gap: 8, lineHeight: 38, accent: C.orange, name: "sim-bullets" });
    addShape(slide, { name: "sim-image-frame", geometry: "roundRect", left: 606, top: 180, width: 620, height: 450, fill: C.panel2, lineFill: C.rule, lineWidth: 1, radius: "rounded-xl" });
    slide.images.add({ blob: await imageBytes("mujoco_vision_right_detection.png"), contentType: "image/png", alt: "MuJoCo 俯视相机橙色方块检测结果", fit: "contain", position: { left: 621, top: 196, width: 590, height: 416 } });
    addText(slide, "画面来自 MuJoCo RGB 渲染，不用物体 body 坐标冒充视觉结果", { name: "sim-caption", left: 606, top: 640, width: 620, height: 30, fontSize: 18, color: C.muted, align: "center" });
    setNotes(slide, "现场先运行 run_mujoco_vision_bridge.bat，再运行 run_mujoco_upper_computer.bat。点击确认后不要只看 Viewer，也要指出上位机相机画面在运动过程中持续更新。", ["outputs/bimanual_task_planner_demo/vision_debug_v18/vision_right_001_detection.png", "upper_computer/docs/MUJOCO_UPPER_INTEGRATION_ZH.md"]);
  }

  // 06 — Simulation evidence, based on slide-17 process timeline.
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.canvas;
    addHeader(slide, "仿真闭环不是静态截图，而是可重复验证的执行过程", 6);
    addShape(slide, { name: "sim-timeline-line", geometry: "straightConnector1", left: 102, top: 344, width: 1018, height: 0, lineFill: C.ink, lineWidth: 2 });
    [102, 508, 914].forEach((x, i) => addShape(slide, { name: `sim-dot-${i}`, geometry: "ellipse", left: x - 8, top: 336, width: 16, height: 16, fill: i === 2 ? C.orange : C.blue }));
    const timeline = [
      { x: 54, label: "视觉确认", title: "RGB → 检测", body: "橙色分割、检测框、像素反投影与目标时效校验" },
      { x: 458, label: "安全与规划", title: "目标 → 任务", body: "工作空间检查、稳定示教点匹配、双臂规划器与碰撞日志" },
      { x: 864, label: "执行与证据", title: "动作 → 结果", body: "抓取、抬升、放置、状态回传、会话与任务日志" },
    ];
    timeline.forEach((t, i) => {
      addText(slide, t.label, { name: `sim-t-label-${i}`, left: t.x, top: 278, width: 220, height: 36, fontSize: 20, color: C.muted, bold: true });
      addText(slide, t.title, { name: `sim-t-title-${i}`, left: t.x, top: 388, width: 300, height: 44, fontSize: 30, bold: true });
      addText(slide, t.body, { name: `sim-t-body-${i}`, left: t.x, top: 454, width: 322, height: 102, fontSize: 21, color: C.muted });
    });
    addText(slide, "2 / 2", { name: "sim-proof-1", left: 54, top: 173, width: 160, height: 64, fontSize: 44, bold: true, color: C.green });
    addText(slide, "原 V18.3 左右视觉放置成功", { name: "sim-proof-1-label", left: 210, top: 188, width: 330, height: 36, fontSize: 21, color: C.muted });
    addText(slide, "51 帧 / 6.06 s", { name: "sim-proof-2", left: 595, top: 173, width: 300, height: 64, fontSize: 44, bold: true, color: C.orange });
    addText(slide, "运动区间 JPEG 全部不同", { name: "sim-proof-2-label", left: 895, top: 188, width: 310, height: 36, fontSize: 21, color: C.muted });
    addText(slide, "当前桥接仍只开放已验证右臂固定点；任意目标和中途急停属于下一阶段。", { name: "sim-boundary", left: 54, top: 607, width: 1130, height: 42, fontSize: 22, bold: true });
    setNotes(slide, "2/2 是原 V18.3 报告结果；51 帧是本次实时推流回归。说明复用了已有可靠动作，同时新增了上位机链路和运动中间帧，而不是重新发明控制器。", ["outputs/bimanual_task_planner_demo/vision_grasp_report_v18.md", "upper_computer/docs/MUJOCO_UPPER_INTEGRATION_ZH.md"]);
  }

  // 07 — Real-side work with evidence images.
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.canvas;
    addHeader(slide, "真机侧已完成“可观察、可诊断、不可误执行”", 7, "REAL-SIDE UPPER COMPUTER / SECOND DEMO");
    addShape(slide, { name: "real-vision-frame", geometry: "roundRect", left: 54, top: 184, width: 432, height: 438, fill: C.panel2, lineFill: C.rule, lineWidth: 1, radius: "rounded-xl" });
    slide.images.add({ blob: await imageBytes("vision_team_status.png"), contentType: "image/png", alt: "视觉组环境与待确认项状态截图", fit: "contain", position: { left: 70, top: 197, width: 400, height: 412 } });
    addText(slide, "视觉接入", { name: "real-vision-title", left: 520, top: 192, width: 260, height: 40, fontSize: 29, bold: true, color: C.blue });
    addBulletList(slide, ["兼容视觉组 robot_state WebSocket", "实时 JPEG、类别、像素中心与深度", "心跳、断线重连、超时和队列限流"], { left: 520, top: 246, width: 390, fontSize: 20, gap: 7, lineHeight: 38, name: "real-v-bullets" });
    addText(slide, "电控接入", { name: "real-elec-title", left: 520, top: 420, width: 260, height: 40, fontSize: 29, bold: true, color: C.orange });
    addBulletList(slide, ["USART10 24 字节编解码与流解析", "J1～J7 CAN ID、路由和限位配置", "串口枚举与 J1 只读监视工具"], { left: 520, top: 474, width: 390, fontSize: 20, gap: 7, lineHeight: 38, accent: C.orange, name: "real-e-bullets" });
    addShape(slide, { name: "real-motor-frame", geometry: "roundRect", left: 938, top: 184, width: 288, height: 438, fill: C.panel2, lineFill: C.rule, lineWidth: 1, radius: "rounded-xl" });
    slide.images.add({ blob: await imageBytes("real_motor_j1_can_tool.png"), contentType: "image/png", alt: "J1 电机 CAN 工具状态截图", fit: "contain", position: { left: 954, top: 198, width: 256, height: 410 } });
    addText(slide, "真机运动写入仍被明确禁用", { name: "real-lock", left: 54, top: 642, width: 1120, height: 32, fontSize: 22, bold: true, color: C.red });
    setNotes(slide, "这里展示的是‘真机侧上位机能力’，不是声称已经完成真机抓取。先展示视觉监视，再展示电控协议和端口诊断，最后指出写控制被安全锁住。", ["upper_computer/report/assets/vision_team_status.png", "upper_computer/vendor/electrical_reference/motor_j1_can_tool.png", "upper_computer/docs/VISION_INTEGRATION_REVIEW_ZH.md", "upper_computer/docs/ELECTRICAL_INTEGRATION_REVIEW_ZH.md"]);
  }

  // 08 — Real-side demo sequence.
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.canvas;
    addHeader(slide, "演示二：真机上位机按“观察 → 诊断 → 只读”展开", 8);
    const steps = [
      { n: "01", title: "连接视觉服务", body: "运行 run_vision_monitor.bat\n接收 D435i / YOLO WebSocket" },
      { n: "02", title: "展示实时感知", body: "JPEG、类别、像素 x/y\n以及深度 z 与失联提示" },
      { n: "03", title: "扫描串口", body: "运行 scan_serial_ports.bat\n证明当前无 USB-UART 候选" },
      { n: "04", title: "展示协议工具", body: "离线编码 / 解码 / 捕获解析\n真机仅允许 J1 只读监视" },
    ];
    // connectors first
    for (let i = 0; i < 3; i++) addShape(slide, { name: `real-demo-arrow-${i}`, geometry: "rightArrow", left: 304 + i * 302, top: 354, width: 58, height: 30, fill: C.blueLight, lineFill: C.blue, lineWidth: 1 });
    steps.forEach((s, i) => {
      const x = 54 + i * 302;
      addShape(slide, { name: `real-demo-step-${i}`, geometry: "roundRect", left: x, top: 242, width: 250, height: 292, fill: i === 3 ? C.orangeLight : C.panel2, lineFill: C.rule, lineWidth: 1, radius: "rounded-xl" });
      addText(slide, s.n, { name: `real-demo-num-${i}`, left: x + 22, top: 270, width: 70, height: 55, fontSize: 38, bold: true, color: i === 3 ? C.orange : C.blue });
      addText(slide, s.title, { name: `real-demo-title-${i}`, left: x + 22, top: 342, width: 206, height: 66, fontSize: 27, bold: true });
      addText(slide, s.body, { name: `real-demo-body-${i}`, left: x + 22, top: 430, width: 206, height: 78, fontSize: 19, color: C.muted });
    });
    addShape(slide, { name: "real-demo-lock-panel", geometry: "roundRect", left: 54, top: 577, width: 1156, height: 74, fill: C.redLight, radius: "rounded-lg" });
    addText(slide, "执行按钮保持锁定：当前演示验证的是上位机接口与安全行为，不是假装已经具备真机闭环。", { name: "real-demo-lock-text", left: 82, top: 598, width: 1095, height: 35, fontSize: 23, bold: true, color: C.red });
    setNotes(slide, "如果现场没有 D435i 或视觉机，用已有消息和只观察界面展示接口；不要为了演示效果连接未知 COM 口。串口扫描是演示的一部分，它证明系统没有猜端口。", ["upper_computer/report/演示与汇报说明.md", "upper_computer/run_course_demo.bat", "upper_computer/tools/list_serial_ports.py"]);
  }

  // 09 — Blocker evidence table, based on Codex Grid slide-14.
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.canvas;
    addHeader(slide, "真机未完成部署：控制闭环与安全依据仍缺失", 9);
    addText(slide, "所有阻塞项都来自现有资料和现场检查；在它们闭环前开放写控制会把软件问题变成人身与设备风险。", { name: "blocker-context", left: 54, top: 168, width: 1140, height: 55, fontSize: 22, color: C.muted });
    const cols = [54, 245, 676, 1000, 1226];
    const headers = ["阻塞项", "当前证据", "缺失闭环", "风险"];
    for (let i = 0; i < 4; i++) {
      addShape(slide, { name: `block-head-${i}`, left: cols[i], top: 246, width: cols[i + 1] - cols[i], height: 54, fill: C.ink, lineFill: C.canvas, lineWidth: 1 });
      addText(slide, headers[i], { name: `block-head-text-${i}`, left: cols[i] + 12, top: 260, width: cols[i + 1] - cols[i] - 24, height: 30, fontSize: 20, color: C.canvas, bold: true });
    }
    const rows = [
      ["视觉标定", "x/y 是像素，z 是深度", "内参、手眼外参、base_link 三维点", "抓错位置"],
      ["固件闭环", "仅回传 J1；pc_data 未驱动电机", "7 关节命令/遥测、ACK、看门狗", "失控或不可验证"],
      ["关节映射", "CAN ID、路由和方向端点已知", "机械零位、符号、偏置与低速实测", "越限 / 碰撞"],
      ["硬件安全", "3.3V TTL、115200 已确认", "USB-UART、物理急停、断电回路", "人员与设备风险"],
    ];
    rows.forEach((row, r) => {
      const y = 300 + r * 74;
      const fill = r % 2 === 0 ? C.panel2 : C.canvas;
      for (let i = 0; i < 4; i++) {
        addShape(slide, { name: `block-cell-${r}-${i}`, left: cols[i], top: y, width: cols[i + 1] - cols[i], height: 74, fill, lineFill: C.rule, lineWidth: 1 });
        addText(slide, row[i], { name: `block-cell-text-${r}-${i}`, left: cols[i] + 10, top: y + 10, width: cols[i + 1] - cols[i] - 20, height: 54, fontSize: i === 0 ? 20 : 18, bold: i === 0, color: i === 3 ? C.red : C.ink });
      }
    });
    addText(slide, "工程判断：保持只读与执行闭锁，是当前资料条件下正确的完成状态。", { name: "blocker-conclusion", left: 54, top: 624, width: 1110, height: 42, fontSize: 25, bold: true, color: C.red });
    setNotes(slide, "不要用‘还没来得及’解释真机未完成。应说明：视觉标定、固件七关节闭环、零位映射、硬急停和适配器都不完整；上位机已经识别这些缺口并阻止误执行。", ["upper_computer/docs/ELECTRICAL_INTEGRATION_REVIEW_ZH.md", "upper_computer/docs/VISION_INTEGRATION_REVIEW_ZH.md", "upper_computer/docs/TEAMMATE_INTERFACE_CHECKLIST_ZH.md"]);
  }

  // 10 — Conclusion and next gates.
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.canvas;
    addHeader(slide, "阶段结论：仿真闭环可验收，真机必须补齐接口后再开放", 10);
    addText(slide, "已交付", { name: "close-left-title", left: 54, top: 190, width: 300, height: 50, fontSize: 32, bold: true, color: C.blue });
    addBulletList(slide, ["统一上位机 GUI、消息、安全和状态机", "MuJoCo 实时视觉与任务执行闭环", "真机视觉监视、USART10 工具与资料审查", "28 项测试、运行日志和演示手册"], { left: 54, top: 266, width: 510, fontSize: 23, gap: 13, lineHeight: 43, name: "close-done" });
    addShape(slide, { name: "close-divider", geometry: "straightConnector1", left: 626, top: 192, width: 0, height: 358, lineFill: C.rule, lineWidth: 1 });
    addText(slide, "真机开放前的四道门槛", { name: "close-right-title", left: 682, top: 190, width: 500, height: 50, fontSize: 32, bold: true, color: C.orange });
    const gates = ["完成 D435i 内参与手眼标定", "升级固件为 7 关节命令/遥测闭环", "低速确认零位、方向、偏置与限位", "验证物理急停、看门狗、ACK 与故障恢复"];
    gates.forEach((g, i) => {
      addText(slide, String(i + 1).padStart(2, "0"), { name: `gate-num-${i}`, left: 682, top: 264 + i * 74, width: 58, height: 44, fontSize: 27, color: C.orange, bold: true });
      addText(slide, g, { name: `gate-text-${i}`, left: 750, top: 267 + i * 74, width: 445, height: 44, fontSize: 22, bold: true });
    });
    addShape(slide, { name: "close-final-panel", geometry: "roundRect", left: 54, top: 586, width: 1156, height: 72, fill: C.blueLight, radius: "rounded-lg" });
    addText(slide, "下一阶段路线：标定 → 固件闭环 → 单关节低速 → 七关节联调 → 视觉抓取", { name: "close-final", left: 80, top: 607, width: 1100, height: 34, fontSize: 25, bold: true, align: "center" });
    setNotes(slide, "收尾回到课程目标：上位机软件工作已经形成可演示、可测试、可继续接真的基础。下一步不是重写界面，而是让视觉与电控按清单补齐可执行证据。", ["upper_computer/report/演示与汇报说明.md", "upper_computer/PROJECT_STRUCTURE_ZH.md", "upper_computer/docs/TEAMMATE_INTERFACE_CHECKLIST_ZH.md"]);
  }

  for (let i = 0; i < presentation.slides.items.length; i++) {
    const slide = presentation.slides.items[i];
    const stem = `slide-${String(i + 1).padStart(2, "0")}`;
    const png = await presentation.export({ slide, format: "png", scale: 1 });
    await fs.writeFile(path.join(RENDERED, `${stem}.png`), new Uint8Array(await png.arrayBuffer()));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(RENDERED, `${stem}.layout.json`), await layout.text());
  }

  const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
  await fs.writeFile(path.join(BUILD, "deck-montage.webp"), new Uint8Array(await montage.arrayBuffer()));
  const inspection = await presentation.inspect({ kind: "slide,textbox,shape,image,notes", maxChars: 30000 });
  await fs.writeFile(path.join(BUILD, "deck-inspect.ndjson"), inspection.ndjson, "utf8");

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(FINAL);
  console.log(`WROTE ${FINAL}`);
  console.log(`SLIDES ${presentation.slides.items.length}`);
}

build().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
