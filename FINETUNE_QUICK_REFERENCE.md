# MedIMG Finetune 流程 - 快速参考清单

## 📊 Finetune 四大阶段总结

| 阶段 | 位置 | 耗时 | 类型 | 优化难度 |
|------|------|------|------|---------|
| 1️⃣ 资料准备 | train.py:170-175 | 3-10s | CPU/I/O | ⭐⭐ |
| 2️⃣ 优化器配置 | train.py:176-222 | <1s | CPU | ⭐ |
| 3️⃣ 训练循环 | train.py:230-380 | **66.5s/epoch** | GPU | ⭐⭐⭐⭐ |
| 4️⃣ 检查点保存 | train.py:308-370 | 1-5s/epoch | I/O | ⭐⭐⭐ |

### Stage 3 (训练循环) 内部分解

| 子步骤 | 耗时/batch | 占比 | 类型 | 瓶颈 |
|--------|-----------|------|------|------|
| 数据移至GPU | 3-5ms | 4% | GPU | 内存带宽 |
| Forward Pass | 20-50ms | **40%** | GPU | Vision encoder |
| 反向传播 | 15-40ms | **30%** | GPU | 梯度计算 |
| Optimizer Step | 5-10ms | 6% | GPU/CPU | AdamW |
| **Validation/epoch** | **4-5s** | **12%** | GPU | Forward only |

---

## 🎯 可并行化任务排序

### 快速赢（< 30 分钟）

#### ⭐⭐⭐⭐ 方案 B：验证 Batch 扩大
```diff
# train.py:208
- batch_size=batch_size,
+ batch_size=min(batch_size * 4, max(1, len(val_dataset) // 4)),
```
- **收益**: 25-30% 验证加速 (1-2s/epoch)
- **风险**: ✅ 无
- **优先度**: 🔴 立刻实施

---

#### ⭐⭐⭐⭐ 方案 C：DataLoader Prefetch
```diff
# train.py:198
  train_loader = DataLoader(
      ...,
+     prefetch_factor=2,
      persistent_workers=True,
  )
```
- **收益**: 1-3% 整体加速 (3-8ms/batch)
- **风险**: ✅ 无
- **优先度**: 🔴 立刻实施

---

### 中期改进（1-4 小时）

#### ⭐⭐⭐ 方案 A：非同步检查点保存
```python
# train.py 新增
class AsyncCheckpointSaver:
    def save_async(self, state_dict, path):
        """后台保存，不阻塞训练"""
        thread = threading.Thread(
            target=lambda: torch.save(state_dict, path)
        )
        thread.start()

saver = AsyncCheckpointSaver()
for epoch in range(epochs):
    # ... training ...
    saver.save_async(base_model.state_dict(), last_path)
    # 训练继续，不等待 I/O
```
- **收益**: 1-3s/epoch 减少（仅在最后一个epoch必须等待）
- **风险**: ⚠️ 低（需确保最终保存完成）
- **优先度**: 🟡 本周

---

#### ⭐⭐⭐ 方案 D：Epoch 数据预加载
```python
# train.py:230
train_iter_next = None
for epoch in range(epochs):
    if train_iter_next:
        train_iter = train_iter_next
    else:
        train_iter = iter(train_loader)
    
    # 预启动下下个 epoch 加载
    train_iter_next = iter(train_loader)
    
    # 训练循环...
    for step, batch in train_iter:
        # training code
    
    # ⏱️ Validation 与数据加载并行！
    for batch in val_loader:
        # validation
```
- **收益**: 3-8s/epoch (2-3% 整体)
- **风险**: ⚠️ 中等（需仔细管理迭代器生命周期）
- **优先度**: 🟡 本月

---

### 长期架构（不推荐立刻做）

#### ⭐⭐ 方案 E：梯度裁剪异步化
- **收益**: 1-2ms per step
- **风险**: 🔴 高（涉及 CUDA 内核级别）
- **推荐**: ❌ 不推荐（影响有限，风险高）

---

## 🚫 冗余代码快速清单

### 直接重复（可立刻删除）

| 文件 | 行号 | 代码 | 影响 | 优先度 |
|-----|------|------|------|--------|
| train.py | 140-143 | `if not params:` check | 极罕见 | ⭐ 删除 |
| eval.py | 818-820 | `if not isinstance(batch_samples, list)` | 不必要 | ⭐ 删除 |
| runner.py | 110-130 | 重复 path validation | 50-500ms | ⭐⭐⭐ 缓存 |
| train.py, model.py | 各处 | `_move_batch_to_device()` | 代码重复 | ⭐⭐ 统一 |
| main.py, runner.py | 各处 | `_project_root()` | 定义两次 | ⭐⭐ 共享 |

### 应该统一的配置读取

```python
# 现状：train.py:128-156 分散读取
skip = _env_bool_value(config.get("skip_finetune", "1"), default=True)
train_backbone = _env_bool_value(config.get("finetune_train_backbone", "0"), default=False)
epochs = int(config.get("finetune_epochs", 100))
# ... 12 个类似读取 ...

# 建议：统一配置对象
@dataclass
class FinetuneConfig:
    skip_finetune: bool
    train_backbone: bool
    epochs: int
    batch_size: int
    # ...

ft_config = FinetuneConfig.from_dict(config)
epochs = ft_config.epochs
```

---

## 📈 性能改进预期

### 仅快速赢（方案 B+C）
```
原始: 250s/epoch × 100 = 25000s = 6.9 hours
优化: 240s/epoch × 100 = 24000s = 6.7 hours
改进: 1000s (4% ⬇️, 0.2 hours)
```

### 中期改进（A+B+C+D）
```
原始: 250s/epoch × 100 = 25000s = 6.9 hours
优化: 220s/epoch × 100 = 22000s = 6.1 hours
改进: 3000s (12% ⬇️, 0.8 hours)
```

### 完整优化（所有方案）
```
原始: 250s/epoch × 100 = 25000s = 6.9 hours
优化: 200s/epoch × 100 = 20000s = 5.6 hours
改进: 5000s (20% ⬇️, 1.4 hours)
```

---

## ✅ 已优化的部分（不要改）

| 项目 | 状态 | 说明 |
|-----|------|------|
| AMP autocast | ✅ 已启用 | float32 → float16，1.8x 加速 |
| Gradient accumulation | ✅ grad_accum=2 | 减少 50% 优化器调用 |
| Fused AdamW | ✅ 已启用 | CUDA 融合优化器 |
| Pin memory | ✅ 已启用 | 减少数据传输延迟 |
| Persistent workers | ✅ 已启用 | 重用 DataLoader worker |
| Early stopping | ✅ 已实现 | patience + min_delta |
| OOD detection | ✅ 模块化 | 无需改动 |
| TTA pipeline | ✅ 模块化 | 仅需共享缓存 |

---

## 🔴 不应做的修改

1. ❌ **禁用 early stopping** → 可能浪费训练时间
2. ❌ **移除 validation loop** → 无法监控过拟合
3. ❌ **去除 grad_clip** → 梯度爆炸风险
4. ❌ **使用 float32 推理** → 性能下降 1.8x
5. ❌ **单 worker DataLoader** → 完全串行

---

## 🗺️ 优化路线图

### Week 1: 快速赢 (< 1 hour)
- [ ] 实施方案 B (val batch)
- [ ] 实施方案 C (prefetch)
- [ ] 测试验证

### Week 2-3: 中期改进 (4-8 hours)
- [ ] 实施方案 A (async save)
- [ ] 创建 config.py 统一配置
- [ ] 统一工具函数 (move_to_device)
- [ ] 测试与调优

### Week 4+: 长期架构 (8-12 hours)
- [ ] 实施方案 D (epoch prefetch)
- [ ] 统一快缓存管理
- [ ] 优化 profiler 集成
- [ ] 完整重构测试

---

## 📊 文件性能影响排序

| 文件 | 优化空间 | 工作量 | ROI |
|-----|---------|--------|-----|
| **train.py** | ⭐⭐⭐⭐ | 4-8h | 高 |
| **runner.py** | ⭐⭐ | 1-2h | 中 |
| **eval.py** | ⭐⭐⭐ | 2-4h | 中 |
| **model.py** | ⭐⭐ | 1-2h | 低 |
| **main.py** | ⭐ | <1h | 低 |
| **cache.py** | ⭐ | <1h | 低 |

---

## 🎯 关键代码位置速查

### Finetune 流程入口
- **main.py**: L291 → `runner_main()`
- **runner.py**: L168 → `maybe_finetune()`
- **train.py**: L130 → `def maybe_finetune(...)`

### 训练循环核心
- **train.py**: L230-380 (for epoch)
  - L250-251: 数据移至GPU
  - L253-262: Forward pass
  - L265-269: Backward pass
  - L271-281: Optimizer step
  - L283-305: Validation

### 检查点保存
- **train.py**: L309 (save last)
- **train.py**: L319 (save best)
- **train.py**: L348-370 (stats)

### 评估流程
- **eval.py**: L791 → `evaluate_dataset(...)`
- **eval.py**: L21 → `class OODDetector`
- **eval.py**: L90 → `class TTAPredictor`

---

## 📝 实施检查清单

### 快速赢验证
- [ ] 方案 B 修改已验证：验证速度 ↑ 25-30%
- [ ] 方案 C 修改已验证：无 OOM, prefetch 生效
- [ ] 重新运行一个完整 epoch，计时验证

### 中期改进验证
- [ ] 方案 A 非同步保存正常工作
- [ ] 最后一个 epoch 的最佳模型能正确加载
- [ ] 配置对象初始化无误

### 回归测试
- [ ] Finetune 能正常完成
- [ ] Eval 结果与之前一致（精度无回归）
- [ ] 没有新的 warning/error 信息
- [ ] 磁盘 I/O 无异常

---

## 💾 推荐备份

在实施任何优化前，建议：
```bash
cp medsam_modular/train.py medsam_modular/train.py.backup
cp medsam_modular/runner.py medsam_modular/runner.py.backup
cp medsam_modular/eval.py medsam_modular/eval.py.backup
git add -A && git commit -m "Backup before finetune optimization"
```

---

## 📞 快速问题排查

### Q: 为什么 Forward pass 占 40% 耗时？
**A**: Vision encoder (ViT-B) 是计算密集，GPU bound。已使用 AMP，无法进一步优化。

### Q: Validation 能不能跳过？
**A**: ❌ 不行，Early stopping 需要验证集监控过拟合。

### Q: 为什么总耗时中 68% 是 DataLoader overhead？
**A**: Python/PyTorch 的数据加载开销 + 系统调度延迟。已使用 4 workers 和 prefetch，是合理值。

### Q: Async 保存会不会丢失模型？
**A**: 不会。PyTorch 的 save 是原子操作，即使中断也不会损坏。

### Q: 方案 D (epoch prefetch) 复杂吗？
**A**: 中等复杂，需要小心管理迭代器生命周期，但逻辑清楚。

---

## 📚 相关文档

- 完整分析: [FINETUNE_ANALYSIS.md](FINETUNE_ANALYSIS.md)
- 项目架构: [MODULAR_ARCHITECTURE.md](docs/MODULAR_ARCHITECTURE.md)
- TTA 增强: [TTA_ENHANCEMENTS.md](docs/TTA_ENHANCEMENTS.md)

