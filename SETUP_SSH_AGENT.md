```
# generate ssh key
ssh-keygen -t ed25519 -C "xxx@gmail.com" -f ~/.ssh/id_rsa_new

# copy values in ~/.ssh/id_rsa_new.pub to github.com ssh key page
```


Setup ~/.ssh/config like below
```
UseKeychain yes
Compression yes
StrictHostKeyChecking no
ForwardAgent yes
ServerAliveInterval 10
LogLevel error
User clcy

Host *
    AddKeysToAgent yes
    IdentityFile ~/.ssh/id_ed25519
    IdentityFile ~/.ssh/id_rsa_new

Host *
  Include /opt/brew/etc/ssh_config
```
Note: If there are multiple identityFiles, it will try one by one. Can be a different ssh key representing different users. 