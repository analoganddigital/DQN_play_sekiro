# Update for using DQN to play sekiro      2021.2.2（English Version）
This is the code of using DQN to play Sekiro .

I am very glad to tell that I have writen the codes of using DQN to play Sekiro .
As is known to all , Supervised learning can only learn skills from the data we provide for it . However , this time by using Reinforcement Learning , we can see a more clever agent playing Sekiro .

Reinforcement Learning can update its network by itself , using the reward feedback , which means we no longer need to collect our own data sets this time . All the data sets come from the real-time interaction between DQN network and the game.
By using this DQN network , you can fight any boss you want in the game .
There still something you need to know ：
* In order to shorten the time of restart the game , we need game modifier .
* The link for game modifier : https://patch.ali213.net/showpatch/118405.html
* See the tutorial video for specific code usage , link : https://www.bilibili.com/video/BV1by4y1n7pe/

Have fun !

# Old version sekiro_tensorflow 
Code link for using Supervised learning to play Sekiro : https://github.com/analoganddigital/sekiro_tensorflow 

Hello everyone , this is analoganddigital . I use this code to complete an interesting porgram of using machine learning to play Sekiro . 
You can see the final presentation in https://www.bilibili.com/video/BV1wC4y1s7oa/ .
I am a junior student in university , which means I can't spend too much time on this program . What a shame ! On the other hand , many audiences hope me share this code . Thus , I eventually put it on the GitHub . 
This is an interesting program , and I hope everyone can enjoy it. In addition , I really welcome you to improve this program , to make this AI more smart !
There still something you need to konw:
* The window size I set is 96*86 , you can change it by yourselves .
* I finally collected 300M training data , if you want better result , maybe you need to collect more data .
* I use Alexnet to finish the training . This program is depend on Supervised learning.
* I have no idea about using Reinforcement learning yet , so I will really appreciate it if someone can help me to overcome this difficulty.(already finished)
* See the tutorial video for specific code usage , link : https://www.bilibili.com/video/BV1bz4y1R7kB

Reference ：
https://github.com/Sentdex/pygta5/blob/master/LICENSE


# 更新——强化学习DQN打只狼     2021.2.2（中文说明）
我非常高兴地告诉大家，我最近又开发出了用DQN强化学习打只狼的代码。
众所周知，监督学习只能学习到我们所提供的数据集的相关技能，但是利用强化学习，我们将看到一个完全不一样的只狼。

强化学习会根据reward奖励进行判断并且自己学习一种打斗方法。更重要的是，我们这次不再需要自己收集数据集了，所有更新数据均来自于DQN网络与游戏的实时交互。
利用这个DQN代码（链接见下方），你可以挑战只狼中任何一个boss，只要boss的血条位置不变即可（因为我采用的是图像抓取的方式获取只狼的血量与boss的血量进行reward判断）。
然后还有一些注意事项：
* 为了缩短只狼复活周期，在这个项目训练中，我们需要采用只狼的24项修改器，让只狼能够原地复活继续训练。
* 修改器下载地址：https://patch.ali213.net/showpatch/118405.html
* 具体代码使用方法请见我在b站上发布的DQN打只狼的教程视频，链接：https://www.bilibili.com/video/BV1by4y1n7pe/

祝各位玩得愉快！

# 旧版本用机器学习打只狼 
旧版本的利用监督学习打只狼的代码链接： https://github.com/analoganddigital/sekiro_tensorflow 。

各位观众大家好，我GitHub用户名是analoganddigital。我用这个程序完成了机器学习打只狼这个项目。
最终效果视频可以看b站https://www.bilibili.com/video/BV1wC4y1s7oa/ 。
我是一个大三学生，真的非常抱歉没能长时间更新这个项目，所以我把它放到了GitHub上面，之前很多观众也是私信我想要代码。
总之我还是希望大家能喜欢这个小项目吧。当然，我非常希望大家能帮忙完善这个程序，万分感激，大家共同讨论我们会获益更多，这其实就是开源的意义。现在由于代码比较基础，所以训练效果不太好。我相信大家会有更多的点子，如果能更新一点算法，我们将会看到一个更机智的AI。我很感谢大家对之前视频的支持（受宠若惊），也十分期待大家有趣的优化，就算没有优化直接用也可以。
还有一些细节我这声明一下：
* 我截取的图像大小是96*86的，各位可以根据自身情况选择。
* 我最终只收集了300M的数据，如果你想训练效果更好的话，可能要收集更多。
* 我用的神经网络是Alexnet，基于监督学习完成的。
* 由于我能力有限，我还没想好如何用强化学习优化算法，所以如果有大佬能分享一下自己的才华，那将十分感谢。（目前已经实现）
* 具体代码使用方法请见我在b站上发布的机器学习打只狼的教程视频，链接： https://www.bilibili.com/video/BV1bz4y1R7kB

部分参考代码：
https://github.com/Sentdex/pygta5/blob/master/LICENSE
