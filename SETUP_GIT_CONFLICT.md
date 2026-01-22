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


## Scenario 2. Local has committed changes, git pull failed

```
# Use git pull --rebase
git pull --rebase

先把你本地的 commit 暂存起来
拉取 remote 的新 commit
把你的 commit 重新 replay 到最新 remote commit 之上
```
