uv run dabench inspect-task task_11 --config configs/react_baseline.local.yaml
uv run dabench run-task task_11 --config configs/react_baseline.local.yaml

```bash
# easy run
cd D:\Project\Hackthon\KDDCUP\baseline\Agent-KDDCup2026
uv run dabench run-and-eval --config configs/react_baseline.qwen.yaml --difficulty easy

# task run
cd D:\Project\Hackthon\KDDCUP\baseline\Agent-KDDCup2026
uv run dabench run-task task_11 --config configs/react_baseline.qwen.yaml
```

```bash
Remove-Item Env:HTTP_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:HTTPS_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:ALL_PROXY -ErrorAction SilentlyContinue
$env:NO_PROXY="localhost,127.0.0.1"
```

```bash
$env:LANGSMITH_API_KEY=""
$env:LANGSMITH_TRACING="true"
$env:LANGSMITH_PROJECT="Trial"
```
