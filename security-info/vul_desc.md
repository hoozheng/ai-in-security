# 漏洞的分类、描述、解决方案

| 漏洞列表 | 漏洞描述 | 问题类型 | 风险等级 | 解决方案 |
| ------- | ------- | ------- | -- | ------- | 
| SQL注入漏洞	| 主要是SQL注入漏洞，对用户提交CGI参数数据未做充分检查过滤，用户提交的数据可能会被用来构造访问后台数据库的SQL指令，从而非授权操作后台的数据库，导致敏感信息泄漏、破坏数据库内容和结构、甚至利用数据库本身的扩展功能控制服务器操作系统。通常在SQL查询语句、LADP查询语句、XPATH查询语句、OS命令，XML解析器、SMTP头、程序参数等中找到。| 代码编程类 | 高 | 使用安全的API,避免使用解释器；对输入的特殊字符进行Escape转义处理；使用白名单来规范化的输入验证方法。sql语句全部使用参数形式调用，不拼sql语句，对输入都要验证：客户端验证+服务器端验证 |  
| 跨站脚本漏洞XSS	| 跨站脚本漏洞，即XSS，通常用Javascript语言描述，利用的是客户端的弱点，常见3种漏洞，1）存储式；2）反射式；3）基于DOM。由于动态网页的web应用对用户提交请求参数未做充分的检查过滤，允许用户在提交的数据中加入HTML、JS代码，未加编码地输出到第三方用户的浏览器，恶意攻击者可以利用Javascript、VBScript、ActiveX、HTML语言甚至Flash应用的漏洞，发送恶意代码给另一个用户，因为浏览器无法识别脚本是否可信，从而跨站漏洞脚本便运行并让攻击者获取其他用户信息。攻击者能盗取会话cookie或session、获取账户、模拟其他用户身份，甚至可以修改网页呈现给其他用户的内容。 | ------- | 高 | 对应用系统源代码进行优化，对用户可控参数，进行严格的后台检测和过滤，对特殊字符进行转义，不能简单的进行JS过滤。对所有web应用输入参数进行过滤，建议过滤出所有以下字符：[1] |（竖线符号）[2] &（&符号）[3];（分号）[4] $（美元符号）[5] %（百分比符号）[6] @（at 符号）[7] '（单引号）[8] "（引号）[9] \'（反斜杠转义单引号）[10] \"（反斜杠转义引号）[11] <>（尖括号）[12] ()（括号）[13] +（加号）[14] CR（回车符，ASCII 0x0d）[15] LF（换行，ASCII 0x0a）[16] ,（逗号）[17] \（反斜杠）或者对特殊字符进行html编码，例如php中的htmlspecialchars()这样的函数。| 






