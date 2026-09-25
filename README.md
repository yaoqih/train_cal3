# train_cal3

福州东调车：确定性物理环境、可组合业务约束、学习策略三层实现。

在固定站场和业务规则下训练一份 GNN 权重；面对新的请求和当前状态，由权重逐勾选择合法动作。支持直接逐勾推演，也支持显式选择有时间预算的学习引导搜索。优化目标是先完成整批任务，再减少**全局总勾数**。训练只使用真实输入请求的组内相对学习，不使用课程生成、教师轨迹或监督预热。CPU 多进程采样，GPU 合批推理与训练。

## 安装与运行

```bash
python -m pip install -e '.[learning,test]'
python -m fzd_shunting train data/point_to_area/augmented --rules configs/dispatch.json \
  --iterations 10000 --workers 8 --device cuda --precision bfloat16 \
  --group-size 8 --groups-per-batch 16 --minibatch-size 128 \
  --trajectory-batch-size 32 --branch-fraction 0.3 \
  --max-hooks 64 --output runs/new-policy.pt

# 用已训练权重输出下一勾；此命令不提交状态变化
python -m fzd_shunting next scenarios/demo.json --rules configs/dispatch.json \
  --checkpoint runs/policy.pt --max-hooks 40 --output runs/next-hook.json

# 连续逐勾推演，再独立回放检查
python -m fzd_shunting plan scenarios/demo.json --rules configs/dispatch.json \
  --checkpoint runs/policy.pt --max-hooks 40 --output runs/demo-plan.json
python -m fzd_shunting replay scenarios/demo.json runs/demo-plan.json \
  --rules configs/dispatch.json
```

需要更高求解质量时，显式使用同一权重引导搜索：

```bash
python -m fzd_shunting plan scenarios/demo.json --rules configs/dispatch.json \
  --checkpoint runs/policy.pt --mode search --seconds 180 --expansions 5000 \
  --max-hooks 40 --workers 4 --device cuda --precision bfloat16 \
  --inference-batch-size 32 --frontier 8000 --states 200000 \
  --output runs/search-plan.json
```

只有 `status=complete` 代表全部终态条件满足。`search_limit`、`cycle_detected`、`dead_end` 可能附带合法部分计划，均不代表完成或已证明无解。搜索保留找到的完整方案，并在预算内尝试减少总勾数；存在分支裁剪，不提供最优性证明。

从已执行状态继续，给 `next` 或 `plan` 传入 `--current-state observed-snapshot.json`，同时仍传入**原始请求**。快照包含物理状态、业务计数和规则／请求指纹，不能用当前车序重建请求来重新识别保护车。`next` 的 `predicted_snapshot` 是预测结果；现场执行之后应以实际观测更新状态。

## 唯一数据约定

```json
{
  "StartStatus": [{
    "No": "example", "Line": "存5线", "Position": 1, "Length": 13.2,
    "RepairProcess": "段修", "Type": "C64K",
    "IsHeavy": false, "IsWeigh": false, "IsClosedDoor": false,
    "TargetLines": {
      "修1库内": {"ForceTargetPosition": [2, 3]},
      "修2库内": {}
    }
  }],
  "TerminalLines": [{"Line": "修1库内", "IsInspectionMode": false}],
  "locoNode": {"Line": "机库线", "End": "North"}
}
```

`TargetLines` 是目标股道映射，多股道任选一个；`ForceTargetPosition` 只位于对应目标股道下面。空配置不限制目标位置。沿当前北→南车序，必须能为各车选出严格递增的允许位置；允许空位，不要求当前连续序号等于目标数字。修库内仍受5／7台位上限约束。

不支持目标字符串数组、车辆顶层位置字段、旧 `Operations` 或 `Request/Response` 格式。不保留旧权重与旧事件的加载分支。计划与事件使用 schema 2；纯相对学习权重使用 schema 4，旧权重直接拒绝。站场配置使用独立的站场格式。未知业务字段直接报错，避免请求约束被忽略。

## 已确认规则

- 一勾只操作一个股道。Get 取北端连续前缀，接到机车尾部；Put 取机车连续后缀，置于股道北端，不倒序。
- 存5南北合并为存5线；洗罐线北、洗罐站合并为洗罐线，洗罐站在南端。修1—修4库内、库外分别操作、分别计勾。
- 取放从北端接近，中间股道有车则阻挡。牵引上限193米，包含15米机车；最多20辆，重车按普通车处理。
- 初始位于修库内、目标含原股道且允许位置含初始台位的车辆为保护车，全程不可移动。有保护车的整条库内禁止缓存，只允许符合终态顺序的放车；其他合法放入的车辆仍可再取出。
- 无保护车的库内可按151.7米临停。终态库内容量按请求迎检模式为5／7辆。临停长度和终停长度分别校核。
- 洗罐油漆北可临停。终点资格由请求负责，规划器不另按线路类别推翻请求；渡线不可停放车辆。
- 支持任意合法初始机车位置与预挂车，结束要求车辆全部送达且机车摘空，不强制回库。
- 四阶段作为经验，不固定为状态机。大库／卸轮的开放勾号通过 [configs/dispatch.json](configs/dispatch.json) 设置。

## 训练与数据

```bash
# 续训已有 schema-4 权重；iterations 表示额外迭代数
python -m fzd_shunting train data/point_to_area/augmented \
  --rules configs/dispatch.json --iterations 10000 --group-size 8 \
  --groups-per-batch 16 --workers 8 --device cuda --precision bfloat16 \
  --minibatch-size 128 --trajectory-batch-size 32 --branch-fraction 0.3 \
  --max-hooks 64 --resume runs/policy.pt --output runs/policy.pt

python -m fzd_shunting audit data/point_to_area/augmented --candidates \
  --rules configs/dispatch.json --output runs/request-audit.json
```

每次迭代同时处理多个请求组，优势只在同一请求、同一起始状态的多条轨迹内计算。约30%的组从真实训练轨迹的中间状态分支，前缀、保护锚点、业务配额和总勾数继续保留；这些分支只更新第一勾，后续推演用于比较长期结果。其余组从原始请求开始，更新全轨迹。保留概率比率裁剪、探索熵和对采样策略的 KL 限制；无价值回归、固定参考模型或成功轨迹模仿。

网络增加股道间全局注意力、取放边界、动作后车序和剩余总勾数预算；动作分布消除“某股道可选数量更多就天然占更多概率”的偏置。每32条轨迹形成一次优化器更新，每128个决策做一次显存分块，默认每迭代最多8次更新。物理／业务规则或图特征契约改变后需要匹配的新权重，不自动改用规则策略。

`iterations` 是迭代次数，每次采样 `groups-per-batch × group-size` 条轨迹。上例每次128条，10000次共128万条。训练会输出吞吐、组内信号、完成率和验证指标；大训练量不自动等于高完成率。GPU 不可用会报错，纯 CPU 运行需显式指定 `--device cpu --precision float32`。

训练总勾数上限默认64，给探索保留空间；验收单独记录**完整解且总勾数≤40**，分支不能重置勾数。不是把每阶段分别限制为40，也不保证请求存在40勾解。

`data/point_to_area/normalized` 与 `augmented` 可直接使用；`raw` 是96份历史来源档案，不是运行时兼容入口。96份增强请求中6份初态超长、4份终态容量分配矛盾、1份已完成；其余85份按日期周分成68份训练、14份验证、3份独立测试。`audit` 的86份通过项包含已完成请求。新增顺序约束是合成数据，必要条件通过不等于已证明可解；容量矛盾不会自动放宽。

需要对新的**规范请求**做增强：

```bash
python -m fzd_shunting prepare-data data/point_to_area/normalized \
  runs/prepared-data --seed 20260924 --fraction 0.2
```

## 查看器与验证

```bash
./run_viewer.sh  # http://127.0.0.1:8502
python -m pytest -q
python scripts/validate_learning.py --checkpoint runs/policy.pt \
  --workers 4 --max-hooks 40 --seconds 120 --output runs/v4-validation.json
```

查看器打开规范请求、schema-2计划或 `{request, plan}` 合集，保留逐勾车列变化、地图／车序表、搜索和离线HTML导出。本机 UI 使用独立 Python 3.10 环境；新环境可安装 `.[ui]`。

物理校验当前覆盖北端接近、拓扑占用、车序、长度与数量。道岔组合、折返清尾、连续跨区段占用，以及机／调合并临停空间仍缺现场参数。局部合法候选也不保证之后一定能完成任务。

独立测试集用于最终报告，不能反复看它来调参。当前默认权重是短程训练的验证产物，尚未获得真实请求完整解；测试通过说明实现与规则契约正常，不代表已达到强求解能力。

详细说明：[架构](docs/architecture.md) · [训练与逐勾使用](docs/learning.md) · [验证结果](docs/validation.md) · [查看器](docs/viewer.md) · [修复前审查记录](docs/three-layer-review.md)。完整率、完整计划总勾数和日期留出验证是长期训练的验收依据。
