import numpy as np
a=[12,4,5,234,56,12,3,46,23,76]

def RS(arr,low,high,k):
    # if low==high:
    #     return a[low]
    i=partition(arr,low,high)
    if k==i-low:
        print(a[k-low])
        return a[k-low]
    if k>i-low:
        RS(arr,i+1,high,k-i)
    else: RS(arr,low,i-1,k)

def partition(arr,low,high): 
    i = ( low-1 )         # 最小元素索引
    pivot = arr[high-1]     
  
    for j in range(low , high): 
        # 当前元素小于或等于 pivot 
        if   arr[j] >= pivot: 
            i = i+1 
            arr[i],arr[j] = arr[j],arr[i] 
  
    arr[i+1],arr[high-1] = arr[high-1],arr[i+1] 
    return i

l=RS(a,0,len(a),4)
print(l)

print(a)
