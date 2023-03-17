import numpy as npd

class Node(object):
    def __init__(self, data):
        # 数据域
        self.data=data
        # 向后的引用域
        self.next = None


class SingleLinkList(object):
    def __init__(self):
        self.head = None

    # is_empty() 链表是否为空
    def is_empty(self):
        return not self.head

    # add(data) 链表头部添加元素
    # O(1)
    def add(self, data):
        node = Node(data)
        node.next = self.head
        self.head = node

    # show() 遍历整个链表
    # O(n)
    def show(self):
        cur = self.head
        while cur != None:
            # cur是一个有效的节点
            print(cur.data, end=' --> ')
            cur = cur.next
        print()

    # append(item) 链表尾部添加元素
    # O(n)
    def append(self, data):
        if self.head==None:
            self.add(data)
            return
        cur = self.head
        while cur.next != None:
            cur = cur.next
        # cur指向的就是尾部节点
        node = Node(data)
        cur.next = node

    # length() 链表长度
    # O(n)
    def length(self):
        count = 0
        cur = self.head
        while cur != None:
            # cur是一个有效的节点
            count += 1
            cur = cur.next
        return count

    # search(item) 查找节点是否存在
    # O(n)
    def search(self, data):
        cur = self.head
        while cur != None:
            if cur.data == data:
                return True
            cur = cur.next
        return False

    # remove(data) 删除节点
    # O(n)
    def remove(self, data):
        cur = self.head
        pre = None
        while cur != None:
            if cur.data == data:
                if cur == self.head:
                    self.head = self.head.next
                    return
                pre.next = cur.next
                return
            pre = cur
            cur = cur.next

    # insert(index, data) 指定位置添加元素
    # O(n)
    def insert(self, index, data):
        if index <= 0:
            self.add(data)
            return
        if index > self.length() - 1:
            self.append(data)
            return
        cur = self.head
        for i in range(index-1):
            cur = cur.next
        # cur指向的是第index节点第前置节点
        node = Node(data)
        node.next = cur.next
        cur.next = node

class Nodearray:
    
    def __init__(self,len):
        self.na=[SingleLinkList()]*len

    def add0(self,data):
        if len(self.na)==0:
            temp=SingleLinkList()
            temp.add(data[1])
            self.na[data[0]].append(temp)
            return True
        else:
            self.na[data[0]].append(data[1])
            return True

    def trv0(self):
        y=[]
        for x in range(len(self.na)):
            cur = self.na[x].head
            count=0
            while cur != None:
                count+=1
                y.append((x,cur.data))
                cur=cur.next
                print("while is executed for",count,"times")
        # return y
        # return len(self.na)
    
    def sch0(self,data):
        if data[0]>=len(self.na):
            print("illegal x")
            return False
        return self.na[data[0]].search(data[1])
                        
    def rmv0(self,data):
        self.na[data[0]].remove(data[1])
        return True
            
