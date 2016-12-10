# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 15:44:34 2016

@author: User
"""

from purifier.purifier import HTMLPurifier
purifier = HTMLPurifier({
    'div': ['*'], # разрешает все атрибуты у тега div
    'span': ['attr-2'], # разрешает только атрибут attr-2 у тега span
    # все остальные теги удаляются, но их содержимое остается
})

#text = open('1CBuh8.hrml', 'r', encoding='utf-8') # for Python 3.5
text = open('1CBuh8-ANSI.hrml', 'r')
for line in text:
    #print(line)
    print purifier.feed(line)