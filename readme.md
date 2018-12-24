# ZhihuLabelsPrediction

**Code for the article "[Automatic Tagging of Zhihu Questions"]()**

## Task Definition
The task aims to tag questions in Zhihu with relevant tags from a collection
of predefined ones. Accurate tags can benefit several downstream applications
such as recommendation and search of Zhihu questions. In this task, you are
challenged to build a multi-label model that assigning relevant tags to a given
question. You will be using a dataset of questions collected from Zhihu’s online
web site.

[More detail](http://tcci.ccf.org.cn/conference/2018/dldoc/taskgline06.pdf)

## Sample input and output
### Input
Given a question: 找房子、租房子一般去哪个网站更好，为什么?

And the question description: 房主儿，58、赶集，搜房、新浪乐居等等哪个更好些？

### Output
We should predict the tag set: {租房网,租房,搜房网}

## How to use

train a model: `python main.py train --model rcnn`

then

test a model: `python main.py test --model rcnn`