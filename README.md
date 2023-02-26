# 电力智能深度分析系统



> 👻 一款基于数据挖掘和深度学习的电力用户分析软件。该软件完成了五项任务，并提供多种内置算法，便于用户选择。

## 相关链接

- 💾项目地址：[点击跳转！](https://gitee.com/fu-ziyang/cup_software/tree/dev/)
- 📘操作指南：[点击观看！](http://cupweb.fcode.ltd/#/guide)
- 🌐软件官网：[官方网址！](http://cupweb.fcode.ltd/#/home)
- 🎬配置视频：[点击观看！](http://cupweb.fcode.ltd/#/config)
- 🎥演示视频：[点击观看！](http://cupweb.fcode.ltd/#/display)

## 环境配置

📝该软件一共分为前端和后端，前端所有框架基于Node搭建，后端基于Flask框架。

❗️❗️❗️<span style="font-weight:bold;color:red;">（注：需要先启动后端服务，再开启前端，以免前端出现接收不到服务的情况）</span>

#### 前端环境搭建

前端我们提供了🔳源代码和🔳可执行文件，请任选一项进行使用，推荐使用<span style="color:red;font-weight:bold;">可执行文件</span>

- 可执行文件：🔗 [下载链接](http://cupweb.fcode.ltd/#/about)
- 源代码

​	✔️前端所有框架均基于node搭建，首先请在电脑上下载并安装🔗 [Node](https://nodejs.org/en/)

​	✔️ 下载或者克隆项目到本地（如果权限不足请下载）🔗备用[下载链接](http://cupweb.fcode.ltd/#/about)

```bash
git clone https://gitee.com/fu-ziyang/cup_software.git
```

​	✔️ 进入项目根目录，打开cmd进入到该目录，使用以下命令安装所有依赖包

```bash
npm install
```

​		如果上述命令无法使用或者网速过慢，请使用以下命令：

```bash
npm install cnpm
cnpm install
```

#### 后端服务搭建

✔️可以使用Python环境或者Anaconda集成环境（推荐使用[Anaconda](https://www.anaconda.com/products/distribution)环境）

​		**1️⃣**下载完成后，运行安装，注意勾选Add to path，表示将安装路径自动添加到系统环境变量中

​		2️⃣打开CMD，运行以下命令查看是否安装成功：

```bash
conda -V
```

​		出现以下输出说明安装成功：

![image-20220706092424803](http://fcode.ltd/image-20220706092424803.png)

​		**3️⃣**新建Python环境，运行以下命令在Anaconda中新建一个Python环境

```bash
conda create -n software python==3.7
```

​		等待一段时间，出现以下输出说明安装成功：

![image-20220706093913143](http://fcode.ltd/image-20220706093913143.png)

​		4️⃣进入以上新建的名为 `software`的python环境

```bash
conda activate software
```

​		☑️至此，后端环境搭建完成！

✔️下载后端服务（不必和前端在一个文件夹，任意一个文件夹均可）🔗[点击下载](http://cupweb.fcode.ltd/#/back)    🔗[备用下载](http://fcode.ltd/back.zip)

✔️进入后端服务根目录，安装所有依赖

```
pip install -r requirements.txt
```

✔️安装完成后，运行根目录下的app.py

```python
python app.py
```

## 项目运行

1️⃣运行后端服务下的app.py并保持其处于运行状态（注意：先激活或者选择合适的Python环境）

```bash
python app.py
```

2️⃣运行可执行文件或者在前端项目的根目录下运行：

```bash
npm run dev
```

等待一段时间，项目会自动构建开发模式下的软件
