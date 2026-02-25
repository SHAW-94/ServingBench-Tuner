# Public workload profiles (production-oriented)

These profiles are **synthetic/public** (no enterprise logs) but intentionally shaped to look like
real LLM serving traffic. Generate the paired replay traces and empirical length distributions with:

```bash
python scripts/make_public_workload.py --out-dir data/workloads --profiles short_qa,long_gen,agent_like --n-requests 400 --seed 42
```

## 1) `short_qa.yaml` — 客服 / FAQ / 在线搜索问答

- **请求长度分布**
  - Prompt 偏短（常见几十到几百 tokens）
  - Output 偏短（几十 tokens 为主）
  - 这类业务的延迟主要被 **排队 + TTFT** 影响，而不是 decode
- **到达模式**
  - 以 replay 为主（底层是公开合成 Poisson + 局部波动）
  - 能复现高峰时段、活动入口、登录峰值的 burst
- **多轮特征**
  - 默认单轮（`sessions.enabled=false`）
  - 更贴近在线客服和搜索问答
- **对应业务场景**
  - 客服机器人、产品 FAQ、检索增强问答（单轮）
- **为什么真实**
  - 真实线上交互大多是短请求，用户对 P95/TTFT 非常敏感；该 profile 能稳定复现
    “平均值还行但尾延迟炸掉”的现象。

## 2) `long_gen.yaml` — 内容生成 / 报告撰写 / 营销文案

- **请求长度分布**
  - Prompt 中等长度（主题 + 指令）
  - Output 明显长尾（几百到几千 tokens）
  - 这类业务更容易暴露 **decode 吞吐 / tok/s / 成本 proxy** 问题
- **到达模式**
  - replay（公开合成）
  - 模拟异步任务或批量生成提交，不追求极低 TTFT，但看总时长和吞吐
- **多轮特征**
  - 默认单轮，重点在长输出
- **对应业务场景**
  - 内容生成、长摘要、代码/文档草稿生成
- **为什么真实**
  - 很多生产业务不是聊天，而是“长输出任务”；这类场景对 batch 策略、max_batch_tokens
    和量化配置的收益最明显。

## 3) `agent_like.yaml` — Agent/RAG 多步任务（强推荐做展示）

- **请求长度分布**
  - Prompt 随轮次增长（上下文累积）
  - Output 波动大（工具调用短回复 + 最终总结长回复混合）
- **到达模式**
  - replay（公开合成）
  - 含 burst/lull，容易在高并发时出现队列拥塞
- **多轮特征**
  - 开启多轮（最高 5 turns）
  - `context_growth: agent_like`，模拟计划→工具→总结链路
- **对应业务场景**
  - Agent 工作流、RAG 多跳问答、工具调用编排
- **为什么真实**
  - 这是最接近生产 Agent 负载的简化版：**长上下文 + 多轮 + 输出长度高波动**，正好会
    触发 P95/P99 尾延迟放大，是展示“质量约束下自动调优”价值的最佳场景。

## 使用建议（面试/演示）

- 先用 `short_qa` 展示在线 SLA 指标（TTFT、P95）。
- 再用 `long_gen` 展示吞吐与 cost proxy 的权衡。
- 最后用 `agent_like` 展示尾延迟归因图、质量护栏和 Pareto 推荐点（最像生产平台）。
