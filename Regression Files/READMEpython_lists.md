
    <h1 style="color:#2E3A87; text-align:center;">Project Analysis</h1>
    <p style="color:#1F6B88; font-size:20px;">This project contains detailed analysis using Jupyter Notebooks. The following sections describe the steps, code implementations, and results.</p>
    <hr style="border: 2px solid #1F6B88;">
    
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # cstu - grayl
# 1. sum all the items in a list
#    for example: list1=[1,2,3,4,5]
list1=(1,2,3,4,5)
print (list1)
sum(list1)
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #2. get the largest number in a list
print ('the list',list1)
max(list1);
print('the largest value is', max(list1 ))
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            
#3. remove duplicate elements froma list
list2=[1,2,2,3,4,5]
print('list with duplicate values', list2) 
my_list2=set(list2) 
print('duplicates removed', my_list2) 
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            
# 4. remove the 0th, 4th, and 5th elements
my_list3 = ['Red','Green','White', 'Black','Pink','Yellow']
#    expected output : ['Green','White','Black')
print('original', my_list3)
my_pop = my_list3.pop(1), my_list3.pop(1), my_list3.pop(1)
print(my_pop)
 
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # 5. return even numbers in a simple list =[1,2,3,4,5]
list1 =[1,2,3,4,5]
print(list1)
list2 = []
def my_even_number(list1):
 for num in list1:
  if num in list1 %2 == 1:
    l = [list1.pop(0)]
    for i in range(list1):
      return l              
#print(my_even_number(n), my_even_number(i))
print('still not working yet!')
 
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #6. create a directory from two lists
my_dir1 =["Black","Red","Maroon","Yellow"],['#000000','#FF0000','#800000','#FFFF00']
my_dir2 = ["Black","Red","Maroon","Yellow"]
my_dir3 = ['#00000','#FF0000','#800000','#FFFF00']
# match the first elements of the remainig list
my_dir4 = my_dir2.pop(0), my_dir3.pop(0)
my_dir5 = my_dir2.pop(0), my_dir3.pop(0)
my_dir6 = my_dir2.pop(0), my_dir3.pop(0)
my_dir7 = my_dir2.pop(0), my_dir3.pop(0)
print('original list:', my_dir1)
print('expected outoput:', my_dir4, my_dir5, my_dir6, my_dir7)
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            from IPython.core.application import List
from IPython.core.interactiveshell import ListType
# 7. Program to do permutations in a list
#    example list: [1,2,3], 
#    Expected output: [1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]
#    Steps: 
#         1. Check to see if list is empty (0)
#         2. If at least one element (1)
#         3. Check length of list
#         4. Do permutation and get remaining list
#         5. CREATE function to test
#   Note: Using online templete to solve this complex problem  

#  ---Permutation Function
#  get number of permutations is range of my_perm

# define permutation function

def permutation(list):    
  if len(list) == 0:
   return []
  if len(list) == 1:
   return [list]
  l = []
  for element in range(len(list)): 
    m = list[element]
    my_rem = list[:element] + list[element + 1:]
    for p in permutation(my_rem):
        l.append(p)
  return l
list = (1,2,3)
for p in permutation(list):
  print(p)
#did it work? why empty -- let me think about this one and come back:
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            from ast import Index

# 8. Program to get difference two lists
#    example list: list1 = [1,2,3,4,5], list2 = [2,3,4,5,6]
#    Difference = [1,6] (one list contains a '1', and one contains a '6')
#    Approach: COMPARE the two list together and REMOVE duplicates 
#    (see activity #3 above)
#    Expected output: [1,6]

my_diff1 =(1,2,3,4,5)
my_diff2 =(2,3,4,5,6)
# decided to look at example online
my_diff3 = []
my_diff4 = []
for element in my_diff1:
  if element not in my_diff2:
    my_diff3.append(element)

for element in my_diff2:
  if element not in my_diff1:
    my_diff4.append(element)
result = [my_diff3, my_diff4]

print('output ==>' , my_diff3, ",", my_diff4)
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #9. Program to determine if two list are circulary identical
# example: list1 = [1,2,3] list2 = [3,2,1]
cir1 = [1,2,3]
cir2 = [3,2,1]  

print('input strings : ',  str(cir1) , str(cir2))
# sort the two strings
cir3 = cir1.sort()
cir4 = cir2.sort()
if cir3 == cir4:  
  print('output: identical')
else:
  print('not identical')
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #10. Program to find second smallest number in a list
#    example list: [1,2,3,4]
#    Expected output:[2] is second smallest number in list
#sm1 = min(sm_list)
#sm2 = max(sm_list)
def my_len(sm_list):
  length = len(sm_list)
  sm_list.sort()
  print('the largest number is: ', sm_list[length -1])
  print('the smallest number is: ', sm_list[0])
  print('the second largest number is:',sm_list[length -2])
  print('the second smallest is: ', sm_list[1])
# Input data
sm_list = [1,2,3,4]
largest = my_len(sm_list)
            </pre>


            
    <hr style="border: 2px solid #1F6B88;">
    <h3 style="color:#2E3A87;">Analysis and Results:</h3>
    <p style="color:#1F6B88; font-size:18px;">The notebook contains various steps for analyzing the dataset. Below you can see the results and analysis conducted during the notebook execution.</p>
    