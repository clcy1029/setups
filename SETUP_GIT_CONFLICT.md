# Tips on solving git conflict


## Scenario 1. Local has uncommitted changes, remote branch is ahead. Can't pull directly

```
# 1. 暂存本地修改
git stash

# 2. 拉取远程更新（使用rebase）
git pull --rebase

# 3. 恢复本地修改
git stash pop

# 4. 解决可能的冲突
# 5. 提交并推送
git push
```