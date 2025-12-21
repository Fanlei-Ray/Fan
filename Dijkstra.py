import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from matplotlib.animation import FuncAnimation

# -------------------------- 核心：中文显示配置（仅用全局设置，不依赖函数参数）--------------------------
# 方案：通过全局配置 matplotlib 字体，让 networkx 绘图自动继承中文字体
try:
    # 1. 优先加载Windows系统自带的微软雅黑字体（无需路径，直接用字体名称）
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    # 2. 兜底：若字体名称无效，直接指定字体文件路径（确保路径存在）
    font_path = r'C:\Windows\Fonts\msyh.ttc'  # 微软雅黑字体文件路径
    font = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font.get_name()
    plt.rcParams['axes.unicode_minus'] = False

# -------------------------- 1. 校园带权无向图模型 --------------------------
campus_graph = {
    '北门': {'逸夫楼': 6, '主楼': 5, '体育场': 3, '一食堂': 4, '9号宿舍楼': 2},
    '逸夫楼': {'北门': 6, '一食堂': 3, '西门': 4, '10号楼': 4, '南门': 5, '主楼': 1, '图书馆': 4},
    '主楼': {'北门': 5, '逸夫楼': 1, '南门': 3, '阶梯教室': 0.5, '文彬楼': 2},
    '体育场': {'北门': 3, '文彬楼': 3},
    '一食堂': {'北门': 4, '9号宿舍楼': 2, '3号楼': 4, '逸夫楼': 3},
    '9号宿舍楼': {'北门': 2, '医务室': 5, '一食堂': 2},
    '文彬楼': {'体育场': 3, '主楼': 2, '文约楼': 0.5},
    '医务室': {'9号宿舍楼': 5, '3号楼': 1},
    '3号楼': {'一食堂': 4, '医务室': 1, '西门': 7},
    '西门': {'逸夫楼': 4, '3号楼': 7, '10号楼': 2},
    '10号楼': {'逸夫楼': 4, '西门': 2, '南门': 6},
    '南门': {'逸夫楼': 5, '主楼': 3, '10号楼': 6, '文正楼': 3},
    '图书馆': {'逸夫楼': 4, '文约楼': 0.5, '文彰楼': 0.5, '阶梯教室': 1},
    '阶梯教室': {'主楼': 0.5, '图书馆': 1},
    '文约楼': {'文彬楼': 0.5, '图书馆': 0.5},
    '文彰楼': {'图书馆': 0.5, '文正楼': 0.5},
    '文正楼': {'文彰楼': 0.5, '南门': 3}
}
all_locations = list(campus_graph.keys())


# -------------------------- 2. 用户交互模块 --------------------------
def user_interaction():
    print("=" * 60)
    print("常州大学科教城校区导航系统")
    print("=" * 60)
    print("可用地点列表（输入编号选择）：")
    for idx, loc in enumerate(all_locations, 1):
        print(f"{idx:2d}. {loc}")
    print("=" * 60)

    # 选择起点
    while True:
        try:
            start_idx = int(input("请输入起点编号：")) - 1
            if 0 <= start_idx < len(all_locations):
                start = all_locations[start_idx]
                break
            else:
                print(f"输入错误！请输入1-{len(all_locations)}之间的编号")
        except ValueError:
            print("输入错误！请输入数字编号")

    # 选择终点
    while True:
        try:
            end_idx = int(input("请输入终点编号：")) - 1
            if 0 <= end_idx < len(all_locations):
                end = all_locations[end_idx]
                break
            else:
                print(f"输入错误！请输入1-{len(all_locations)}之间的编号")
        except ValueError:
            print("输入错误！请输入数字编号")

    if start == end:
        print(f"\n✅ 起点和终点均为【{start}】，无需导航！")
        return None, None

    print(f"\n📌 您选择的路线：{start} → {end}")
    return start, end


# -------------------------- 3. Dijkstra算法（记录运算过程）--------------------------
def dijkstra_with_process(graph, start):
    dist = {vertex: float('inf') for vertex in graph}
    dist[start] = 0
    visited = set()
    prev = {vertex: None for vertex in graph}
    process_log = []

    # 初始状态记录
    process_log.append((set(visited), dist.copy(), None))

    while len(visited) < len(graph):
        # 选取未访问的最小距离顶点
        unvisited = [v for v in graph if v not in visited]
        current_vertex = min(unvisited, key=lambda x: dist[x])

        # 记录当前步骤
        process_log.append((set(visited), dist.copy(), current_vertex))

        # 标记已访问
        visited.add(current_vertex)

        # 更新邻接顶点距离
        for neighbor, weight in graph[current_vertex].items():
            if neighbor not in visited:
                new_dist = dist[current_vertex] + weight
                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    prev[neighbor] = current_vertex

    # 最终状态记录
    process_log.append((set(visited), dist.copy(), None))
    return dist, prev, process_log


# -------------------------- 4. 最短路径回溯 --------------------------
def get_shortest_path(prev, start, end):
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = prev[current]
        # 防止异常循环
        if len(path) > len(prev) + 1:
            return None
    path.reverse()
    return path if path[0] == start else None


# -------------------------- 5. 可视化模块--------------------------
def visualize_dijkstra(process_log, graph, start, end, shortest_path):
    # 1. 构建NetworkX图
    G = nx.Graph()
    G.add_nodes_from(graph.keys())
    for u in graph:
        for v, w in graph[u].items():
            G.add_edge(u, v, weight=w)

    # 2. 固定节点布局（确保每次运行位置一致）
    pos = nx.spring_layout(G, seed=42, k=2.8)  # seed=42保证布局固定

    # 3. 创建画布
    fig, ax = plt.subplots(figsize=(16, 12))

    # 4. 绘制总标题（用matplotlib原生函数，确保中文）
    fig.suptitle(f'Dijkstra算法运算过程可视化\n起点：{start} → 终点：{end}',
                 fontsize=18, fontweight='bold', y=0.95)

    # 5. 颜色配置
    color_map = {
        'unvisited': '#87CEEB',  # 浅蓝（未访问）
        'visited': '#FF6B6B',  # 红色（已访问）
        'current': '#FFA500',  # 橙色（当前选中）
        'start': '#32CD32',  # 绿色（起点）
        'end': '#9370DB'  # 紫色（终点）
    }

    # 6. 绘制初始元素
    # 6.1 绘制边（灰色普通边）
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#CCCCCC', width=1.8, alpha=0.7)

    # 6.2 绘制边权值（无fontproperties，依赖全局字体配置）
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                 font_size=9, label_pos=0.3)  # 无fontproperties

    # 6.3 初始化节点颜色和标签
    init_colors = [color_map['unvisited'] for _ in G.nodes()]
    init_labels = {node: f"{node}\n∞" for node in G.nodes()}
    init_labels[start] = f"{start}\n0"  # 起点距离为0

    # 6.4 绘制节点（无字体参数）
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=3500, node_color=init_colors,
                                   edgecolors='black', linewidths=2.5)

    # 6.5 绘制节点标签
    labels = nx.draw_networkx_labels(G, pos, ax=ax, labels=init_labels,
                                     font_size=10, font_weight='bold')

    # 7. 绘制图例（用matplotlib原生函数，确保中文）
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color_map['unvisited'], label='未访问节点'),
        plt.Rectangle((0, 0), 1, 1, facecolor=color_map['visited'], label='已访问节点'),
        plt.Rectangle((0, 0), 1, 1, facecolor=color_map['current'], label='当前选中节点'),
        plt.Rectangle((0, 0), 1, 1, facecolor=color_map['start'], label='起点'),
        plt.Rectangle((0, 0), 1, 1, facecolor=color_map['end'], label='终点'),
        plt.Line2D([0], [0], color='#0000FF', linewidth=3, label='最短路径边')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)

    # 8. 调整画布范围
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.axis('off')  # 隐藏坐标轴

    # 9. 动画更新函数
    def update(frame):
        nonlocal nodes, labels
        visited_set, dist_dict, current_node = process_log[frame]

        # 9.1 更新节点颜色
        new_colors = []
        for node in G.nodes():
            if node == start:
                new_colors.append(color_map['start'])
            elif node == end:
                new_colors.append(color_map['end'])
            elif node == current_node:
                new_colors.append(color_map['current'])
            elif node in visited_set:
                new_colors.append(color_map['visited'])
            else:
                new_colors.append(color_map['unvisited'])

        # 9.2 更新节点标签
        new_labels = {}
        for node in G.nodes():
            dist_val = dist_dict[node]
            dist_str = f"{dist_val:.1f}" if dist_val != float('inf') else "∞"
            new_labels[node] = f"{node}\n{dist_str}"

        # 9.3 清除原有元素并重新绘制
        nodes.remove()
        for text in labels.values():
            text.remove()

        # 重新绘制节点
        nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=3500, node_color=new_colors,
                                       edgecolors='black', linewidths=2.5)

        # 重新绘制节点标签（无fontproperties）
        labels = nx.draw_networkx_labels(G, pos, ax=ax, labels=new_labels,
                                         font_size=10, font_weight='bold')

        # 9.4 最后一帧绘制最短路径
        if frame == len(process_log) - 1 and shortest_path:
            path_edges = [(shortest_path[i], shortest_path[i + 1]) for i in range(len(shortest_path) - 1)]
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=path_edges,
                                   edge_color='#0000FF', width=4, alpha=0.9)
            # 显示最终结果标题
            total_time = dist_dict[end]
            ax.set_title(f'✅ 算法结束！最短路径：{" → ".join(shortest_path)} | 总步行时间：{total_time:.1f}分钟',
                         fontsize=14, pad=20)
        else:
            # 显示当前步骤标题
            step_text = f'第{frame}步：{"初始状态" if current_node is None else f"选中节点【{current_node}】，更新邻接节点距离"}'
            ax.set_title(step_text, fontsize=12, pad=20)

        return nodes, labels

    # 10. 创建动画（每1.8秒一帧，不重复）
    ani = FuncAnimation(fig, update, frames=len(process_log), interval=1800,
                        blit=False, repeat=False)

    # 11. 显示画布
    plt.tight_layout()
    plt.show()


# -------------------------- 6. 主函数 --------------------------
def main():
    # 1. 用户选择路线
    start, end = user_interaction()
    if start is None or end is None:
        return

    # 2. 计算最短路径
    print("\n🔍 正在计算最短路径...")
    dist, prev, process_log = dijkstra_with_process(campus_graph, start)

    # 3. 回溯路径
    shortest_path = get_shortest_path(prev, start, end)

    # 4. 输出结果
    print("=" * 60)
    if shortest_path:
        total_time = dist[end]
        print(f"📊 导航结果：")
        print(f"   最短路径：{' → '.join(shortest_path)}")
        print(f"   总步行时间：{total_time:.1f}分钟")
    else:
        print(f"❌ 抱歉，未找到从【{start}】到【{end}】的可达路径！")
    print("=" * 60)

    # 5. 可视化
    if shortest_path:
        print("\n🖼️ 正在加载可视化窗口...（关闭窗口可结束程序）")
        visualize_dijkstra(process_log, campus_graph, start, end, shortest_path)


# -------------------------- 运行程序 --------------------------
if __name__ == "__main__":
    # 检查依赖包
    try:
        import networkx
        import matplotlib
    except ImportError:
        print("⚠️ 检测到缺少依赖包，请先执行以下命令安装：")
        print("pip install networkx matplotlib numpy")
        exit()
    # 启动主程序
    main()