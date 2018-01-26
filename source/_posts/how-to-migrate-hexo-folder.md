---
title: （转载）如何在不同终端之间迁移HEXO博客文件夹
date: 2018-01-25 20:41:33
tags: [HEXO教程，HEXO迁移]
---
# 旧终端的操作
>在**username.github.io**仓库中，目前默认的分支是master,用来存放静态文件。我们新建一个分支hexo,并设为默认分支，用来存放环境文件。
<!-- more -->

>这里是比较关键的一步了，hexo分支是基于master分支的，它在创建时被赋予了与master分支同样的内容--静态文件。做乜啊，hexo分支不是要存放环境文件的吗？我当时就是在这里比较晕，因为知乎的高票答案在这里说的并不是很清楚。其实现在目的很清晰：**我们希望hexo分支是空的，并把我们本地的hexo项目与hexo分支关联起来，好把环境文件推送过去。**要达到这个目的，要分以下几步走：

+ github上切换到hexo分支，`git clone`仓库到本地。
+ 此时本地会多出一个username.github.io文件夹，命令行cd进去，删除除.git文件夹（如果你看不到这个文件夹，说明是隐藏了。windows下需要右击文件夹内空白处，点选'显示/隐藏 异常文件'，Mac下我就不知道了）外的其他文件夹。
+ 命令行`git add -A`把工作区的变化（包括已删除的文件）提交到暂存区（ps:`git add .`提交的变化不包括已删除的文件）。
+ 命令行`git commint -m "some description"`提交。
+ 命令行`git push origin `hexo推送到远程hexo分支。此时刷下github，如果正常操作，hexo分支应该已经被清空了。
+ 复制本地username.github.io文件夹中的.git文件夹到hexo项目根目录下。此时，hexo项目已经变成了和远程hexo分支关联的本地仓库了。而username.github.io文件夹的使命到此为止，你可以把它删掉，因为我们只是把它作为一个“中转站”的角色。以后每次发布新文章或修改网站样式文件时，`git add . ` & `git commit -m "some description"` & `git push origin hexo`即可把环境文件推送到hexo分支。然后再`hexo g -d`发布网站并推送静态文件到master分支。

>至此，hexo的环境文件已经全部托管在github的hexo分支了。

作者：nikolausliu
链接：https://www.jianshu.com/p/fceaf373d797
來源：简书
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

## 作者没有注意的一点：

以上步骤过后，**如果hexo使用的主题是`git clone`命令得到的**，那么还需要做一下几个步骤：

1. 看下next主题文件夹下是否有.git文件夹和.gitignore文件，这两个文件存在的话git提交是会忽略掉next主题的。
2. 删除后因为该文件夹已经被纳入了版本管理中，先清空该文件夹的本地缓存然后再添加就行了
`git rm -r --cached path`
3. `git add -A`，`git commit -m "some description"`，`git push origin hexo`

#新终端的操作
##在新终端中git clone成功后，本地出现myhexo文件夹，开始安装hexo环境

1. `cd myhexo`
2. `npm install hexo`
3. `npm install`
4. `npm install hexo-deployer-git`

##在新终端中继续使用和管理hexo

###新建文章并修改

`hexo new xxx`

###提交原始md文件至远程repo

1. `git status`
2. `git add`
3. `git commit -m xx`
4. `git push`

###发布静态网页

hexo g -d

##切换至旧终端使用hexo

###更新网站原始文件

1. `git pull`
2. `hexo g`
3. `hexo server`

#2018/1/26
更详细的可以参考这个教程：**[Hexo 换终端/换电脑小记](http://wangmuduo.com/2016/04/02/hexo-change-os/)**