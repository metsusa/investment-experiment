import Node as nd

#------以下是关于单链表的测试--------
# a=nd.SingleLinkList()
# a.show()    #1
# a.append(12)
# a.show()    #2
# a.add(10)
# a.show()    #3
# a.remove(11)
# a.show()    #4
# a.remove(12)
# a.show()      #5
# a.insert(6,1)
# a.show()    #6
# a.insert(2,4)
# a.show()    #7
#----------------------------

#--------------以下是关于链表数组的测试----------
b=nd.Nodearray(10)
for i in range(0,10,1):
    for j in range(1,5,1):
        b.add0([i,j])
r_y=b.trv0()
print(r_y)      #1
b.rmv0([1,3])
r_y=b.trv0()
print(r_y)      #2
flag=b.sch0([1,3])
print(flag)     #3
flag=b.sch0([1,6])
print(flag)     #4
flag=b.sch0([1,4])
print(flag)     #5
flag=b.sch0([11,4])
print(flag)     #6
