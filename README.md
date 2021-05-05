# 我的博客
## 仓库简介
本仓库存放了我的个人网站 https://tangshusen.me/ 的相关文件，含有两个branch: master和hexo(default):
* 默认分支hexo存放相关hexo文件(即我本地的blog文件夹所有内容)如md文件等等;
* master存放发布到网站的由md文件生成的静态文件，即md博客文件写好后通过命令`hexo g -d`发布网站时会将静态文件push到master分支。


## 写文章步骤

1. 确保本地仓库是切换到了hexo分支并pull一下；
2. 执行`hexo n "blog_title"`新建文章，可以发现`source/_posts/`目录多了一个文件夹和一个.md文件，分别用来存放文章图片和文章内容；
3. 编辑、撰写文章或做其他博客更新改动；
4. `hexo g`生成静态网页，然后`hexo s`本地预览效果，确保没问题后执行`hexo d`（在此之前，有时可能需要执行hexo clean）部署，会自动将最新静态文件改动更新到master分支了；
5. 然后将本地hexo分支的改动也更新到git。


## hexo常用命令
`npm install hexo -g` # 安装Hexo
`npm update hexo -g` # 升级
`hexo init` # 初始化博客


`hexo n "blog_title"` == `hexo new "blog_title"` # 新建文章
`hexo g` == `hexo generate` # 生成
`hexo s` == `hexo server` # 预览
`hexo d` == `hexo deploy` # 部署
`hexo clean` # 清除缓存,若是网页正常情况下可以忽略这条命令

## 参考文章
[GitHub+Hexo 搭建个人网站详细教程](https://zhuanlan.zhihu.com/p/26625249)  
[超详细Hexo+Github博客搭建小白教程](https://godweiyang.com/2018/04/13/hexo-blog/)  
[利用Hexo在多台电脑上提交和更新github pages博客](https://www.jianshu.com/p/0b1fccce74e0)     
[hexo博客同步管理及迁移](https://www.jianshu.com/p/fceaf373d797)     
[Mathjax公式快速参考](https://colobu.com/2014/08/17/MathJax-quick-reference/)   
[各种图标](https://fontawesome.com/v4.7.0/icons/)   
[Hexo个人博客站点被百度谷歌收录](https://blog.csdn.net/qq_32454537/article/details/79482914)      
[Add https to your Namecheap Domain hosted on Github Pages](https://medium.com/@goelanirudh/add-https-to-your-namecheap-domain-hosted-on-github-pages-d66fd96308b5)      
[hexo 主题优化](https://keung.asia/posts/17051/)


### 文章访问量统计
[Leancloud访客统计插件重大安全漏洞修复指南](https://leaferx.online/2018/02/11/lc-security/)       
[关于Hexo在NexT主题更新到6.0+之后出现的有关Leancloud的问题](https://hexawater.ink/2018/11/15/About-Leancloud/)

## TODO
使用第三方Valine完善评论邮件提醒：[Hexo-NexT 配置 Valine](https://tding.top/archives/ed8b904f/)
