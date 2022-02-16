## 第二步，npm 安装omniboard

npm install -g omniboard

## 第三步，开启omniboard服务。平时也是用该命令开启omniboard可视化前端

# 开启用法
omniboard -m hostname:port:database

# 默认情况下如下，其中27017是MongoDB的端口
omniboard -m localhost:27017:appai_sacred

## 第四步，打开 http://localhost:9000 来查看前端，并进行管理。